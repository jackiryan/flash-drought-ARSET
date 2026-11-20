"""Flash drought case study: Western Oklahoma, 2026 (SWDI only).

Repeats the SWDI part of oklahoma_2024.py for 2026 over the same region of Western
Oklahoma (34.00°-36.00°N, 98.00°-100.00°W). GOSIF isn't available for 2026, so the
SIF-RCI and SIF Z-Score are left out.

The hosted SMAP L4 virtual dataset read by the notebooks ends in March 2025. Instead, a
virtual dataset of the 2026 SMAP L4 granules is built as shown in 3_appendix.ipynb, and
the SWDI is computed from it in place of the hosted one:

1. Virtualize the 2026 SPL4SMGP granules, saving the references to inputs/.
2. Compute the SWDI time series from them, saving the per-step rasters.
3. Apply the per-grid-cell flash drought detection rule.
4. Plot the time series, the detections and maps of the SWDI, and animate it as a GIF.

The field capacity and wilting point are not derived again. They describe the soil, so
they're estimated from the whole 2015-2025 record, which one year of data is too short
to stand in for. The ones cached for the region by oklahoma_2024.py are used instead.

Run from the notebooks directory:
    python oklahoma_2026.py

Building the virtual dataset is slow the first time (about 75 minutes for the 2,569
granules from January 1 to November 17, 2026). It covers the granules released when it's
built, so delete it to build it again with the newer ones.
"""
import contextlib
import hashlib
import os
import warnings
from collections.abc import Sequence
from datetime import datetime

import earthaccess
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.patches import Rectangle

import oklahoma_2024 as ok
from analysis import compute_swdi_timeseries, detect_flash_drought_swdi

year = 2026
date_range = [datetime(year, 1, 1), datetime(year, 12, 31)] # noqa: DTZ001
start, end = date_range
year_str = str(year)

swdi_time_series_fname = f"{ok.region_name}_{year_str}_swdi.csv"
swdi_csv_path = f"data/{swdi_time_series_fname}"
raster_dir = f"data/{ok.region_name}_{year_str}_time_steps"
fd_dir = f"data/{ok.region_name}_{year_str}_fd"

# The virtual dataset of the year's SMAP L4 granules, used in place of the hosted one
vds_path = f"inputs/SPL4SMGP_{year_str}_virtual_https.parquet"

ts_fig_path = f"inputs/WOklahoma_SWDI_{year_str}.png"
detection_fig_path = f"inputs/WOklahoma_FD_detection_{year_str}.png"
map_fig_path = f"inputs/WOklahoma_FD_maps_{year_str}.png"
gif_path = f"inputs/WOklahoma_SWDI_{year_str}.gif"


"""I. Inputs"""


def collapse_time_to_scalar(ds: xr.Dataset) -> xr.Dataset:
    """Collapse a SMAP L4 granule's single timestamp to a scalar coordinate.

    Each SMAP L4 granule carries one timestamp stored as a length-1 time
    variable along its own (non-indexed) dimension rather than as a scalar.
    When open_virtual_mfdataset concatenates with concat_dim="time",
    xarray tries to expand_dims("time") to create the stacking dimension
    and raises "time already exists as coordinate or variable name" because
    a non-scalar time variable is already present.

    Squeezing the length-1 dimension turns time into a scalar coordinate,
    which expand_dims is able to promote into a time dimension.
    """
    if "time" in ds.variables:
        squeeze_dims = [d for d in ds["time"].dims if ds.sizes[d] == 1]
        if squeeze_dims:
            ds = ds.squeeze(squeeze_dims)
        ds = ds.set_coords("time")
    return ds


def virtualize_smap_l4(
    temporal: tuple[str, str],
    group: str = "Geophysical_Data",
    variables: Sequence[str] | None = ("sm_rootzone",),
    load: bool = False,
) -> xr.Dataset:
    """Build a virtual dataset of SMAP L4 geophysical variables over a time range.

    This is the function from 3_appendix.ipynb. SMAP L4 (SPL4SMGP) stores the
    coordinate variables (time, x, y) in the file's root group, while the geophysical
    variables such as sm_rootzone live in the Geophysical_Data subgroup as 2-D (y, x)
    arrays with no time coordinate of their own. Because open_virtual_mfdataset parses
    one group per call, the root group and the geophysical group are opened separately
    and the root coordinates are then attached to the stacked geophysical variables.

    Arguments:
        temporal (tuple[str, str]): (start, end) date strings passed to earthaccess.
        group (str): The subgroup holding the geophysical variables.
        variables (Sequence[str], optional): Variables to keep from group. None keeps the whole
            group, the default keeps only sm_rootzone.
        load (bool, optional): When False (default) the geophysical variables stay as
            VirtualiZarr ManifestArrays (byte-range references), which is what writing
            the kerchunk references needs. When True they are materialised into a
            concrete, lazily-loaded dataset that can be computed on directly.

    Returns:
        A virtual xarray.Dataset (or a concrete one when load=True) with the requested
        variables stacked along a time dimension and carrying the time/x/y coordinates.
    """
    auth = earthaccess.login()
    if not auth.authenticated:
        auth.login(strategy="interactive", persist=True)

    warnings.filterwarnings("ignore", "As of version 1.0*", FutureWarning)
    results = earthaccess.search_data(
        short_name="SPL4SMGP",
        temporal=temporal,
    )

    open_options = {
        "access": "indirect",
        "concat_dim": "time",
        "coords": "minimal",
        "compat": "override",
        "combine_attrs": "override",
    }

    warnings.filterwarnings(
        "ignore",
        message="This DMRpp contains the variable EASE2_global_projection*",
        category=UserWarning
    )
    # The root group is always loaded so that time/x/y come back as concrete
    # coordinates.
    result_root = earthaccess.virtualize(
        granules=results,
        load=True,
        data_vars="minimal",
        preprocess=collapse_time_to_scalar,
        loadable_variables=["time", "x", "y"],
        **open_options, # type: ignore
    )

    result_gph = earthaccess.virtualize(
        granules=results,
        load=load,
        group=group,
        data_vars="all",
        **open_options, # type: ignore
    )

    result = result_gph.assign_coords(
        time=result_root["time"],
        x=result_root["x"],
        y=result_root["y"],
    )

    if variables is not None:
        result = result[list(variables)]

    return result


def build_smap_vds() -> str:
    """Virtualize the year's SMAP L4 granules as in 3_appendix.ipynb, if not done already.

    Returns:
        str: Path of the kerchunk (parquet) references of the virtual dataset.
    """
    if not os.path.exists(vds_path):
        import virtualizarr  # noqa: F401  (registers the .vz dataset accessor)

        ds = virtualize_smap_l4((start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")))
        ds.vz.to_kerchunk(vds_path, format="parquet")
        print(f"SWDI: Saved virtual Zarr references for {ds.sizes['time']} granules to {vds_path}")
    return vds_path


def file_digest(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def compute_swdi() -> None:
    if os.path.exists(swdi_csv_path):
        return

    # compute_swdi_timeseries only reads the cached field capacity and wilting point if
    # they match the region. Otherwise it derives them from the whole record of the
    # virtual dataset it's given, which here is only the one year.
    if not os.path.exists(ok.fc_wp_path):
        msg = (f"{ok.fc_wp_path} is missing. It's derived from the 2015-2025 record by "
               "oklahoma_2024.py, or the field capacity and wilting point section of 3_appendix.ipynb.")
        raise FileNotFoundError(msg)
    fc_wp_digest = file_digest(ok.fc_wp_path)

    compute_swdi_timeseries(
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
        swdi_time_series_fname,
        (ok.west, ok.south, ok.east, ok.north),
        ref_url=build_smap_vds(),
        raster_dir=raster_dir,
        fc_wp_path=ok.fc_wp_path,
    )

    if file_digest(ok.fc_wp_path) != fc_wp_digest:
        os.remove(swdi_csv_path)
        msg = (f"The field capacity and wilting point in {ok.fc_wp_path} were derived again from "
               f"{year_str} alone. Restore them with git checkout and run this again.")
        raise RuntimeError(msg)
    print(f"SWDI: Saved time series values to {swdi_csv_path}")


"""II. Plots"""


def format_month_axis(fig: plt.Figure, ax: plt.Axes) -> None:
    ax.xaxis.set_major_formatter(ok.MonthYearFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.set_xlim(start, end) # type: ignore
    fig.autofmt_xdate()
    fig.tight_layout()


def plot_time_series() -> None:
    """The SWDI panel of the time series figure of oklahoma_2024.py."""
    dates, swdi = ok.read_time_series(swdi_csv_path, "time", ["swdi"])

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ok.plot_swdi_series(ax, dates, swdi["swdi"], f"SWDI over {ok.plot_region_name} ({year_str})\n{ok.coords_str}",
                        label_crops=True)
    format_month_axis(fig, ax)
    fig.savefig(ts_fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved {ts_fig_path}")


def plot_detection() -> None:
    """The SWDI panel of the flash drought detection figure of oklahoma_2024.py."""
    dates, swdi = ok.read_time_series(swdi_csv_path, "time", ["swdi", "fd_percent"])

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ok.plot_swdi_detection(ax, dates, swdi["swdi"], swdi["fd_percent"],
                           f"SWDI with Flash Drought Detections over {ok.plot_region_name} ({year_str})\n"
                           f"{ok.coords_str}", label_crops=True)
    format_month_axis(fig, ax)
    fig.savefig(detection_fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved {detection_fig_path}")


def plot_maps(map_dates: list[tuple[datetime, str]]) -> None:
    """Small multiples of the SWDI per grid cell with detections hatched, as in the 2024 maps."""
    swdi_dates, _ = ok.read_time_series(swdi_csv_path, "time", [])

    n_cols = len(map_dates)
    fig, axes = plt.subplots(1, n_cols, figsize=(2.15 * n_cols + 1.0, 3.9), squeeze=False, layout="constrained")
    for col, (map_date, label) in enumerate(map_dates):
        swdi_date = min(swdi_dates, key=lambda d: abs(d - map_date))
        ax = axes[0, col]
        mesh = ok.draw_swdi_map(ax, swdi_date, raster_dir, fd_dir, label_towns=col == 0)
        ax.set_title(f"{label}\n{swdi_date:%b %d}", fontsize=11)

    fig.colorbar(mesh, ax=axes[0, :].tolist(), label="SWDI", extend="both", shrink=0.85)
    hatch_handle = Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="black", linewidth=0.0, hatch="///")
    fig.legend(
        [hatch_handle],
        [f"Flash drought detected in grid cell (SWDI fell by 2+ over 30 days to {ok.swdi_thresh} or below)"],
        loc="outside lower center", frameon=False, fontsize=10,
    )
    fig.suptitle(f"SWDI over {ok.plot_region_name} in {year_str}\n{ok.coords_str}", fontsize=13)
    fig.savefig(map_fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {map_fig_path}")


def main() -> None:
    compute_swdi()
    detect_flash_drought_swdi(raster_dir, fd_dir, swdi_csv_path, abs_threshold=ok.swdi_thresh)
    plot_time_series()
    plot_detection()
    # The phases of the year, following the time series of SWDI. The region entered the
    # year in drought, so the detections mark it drying further rather than drying out
    # from normal conditions.
    plot_maps([
        (datetime(year, 1, 1), "Winter drought"), # noqa: DTZ001
        (datetime(year, 5, 16), "Spring drying"), # noqa: DTZ001
        (datetime(year, 6, 27), "June rains"), # noqa: DTZ001
        (datetime(year, 7, 6), "Summer onset"), # noqa: DTZ001
        (datetime(year, 9, 7), "Peak severity"), # noqa: DTZ001
        (datetime(year, 10, 16), "October rains"), # noqa: DTZ001
    ])
    ok.animate_swdi(gif_path, swdi_csv_path, raster_dir, fd_dir)


if __name__ == "__main__":
    with contextlib.chdir(os.path.dirname(os.path.abspath(__file__))):
        main()
