"""Flash drought case study: Western Oklahoma, 2024.

Applies the workflow from 2_detection.ipynb to a region of Western Oklahoma
(34.00°-36.00°N, 98.00°-100.00°W):

1. Build the per-grid-cell SIF climatology for the region (as in 3_appendix.ipynb).
2. Compute the SIF-RCI and SWDI time series, saving the per-step rasters.
3. Apply the per-grid-cell flash drought detection rules.
4. Plot the time series, the detections and maps of the drought's development.
5. Animate the per-cell SIF-RCI, SIF Z-Score and SWDI over the year as GIFs.

Run from the notebooks directory:
    python oklahoma_2024.py

The SWDI step derives field capacity and wilting point from the full 2015-2025 SMAP
L4 record over the region, which is slow (hours) the first time. The result is cached
to inputs/oklahoma_fc_wp.tif so later runs skip it.
"""
import base64
import contextlib
import csv
import functools
import glob
import json
import os
import re
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import rasterio
from matplotlib import patheffects
from matplotlib.collections import Collection, PatchCollection, QuadMesh
from matplotlib.colors import TwoSlopeNorm, to_rgba
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path
from PIL import Image
from pyproj import Transformer

from analysis import (
    EASE2_GLOBAL_EPSG,
    compute_sif_time_series,
    compute_swdi_timeseries,
    detect_flash_drought_sif,
    detect_flash_drought_swdi,
    get_read_window,
    read_roi,
)
from download import download_unpack_gosif

# Western Oklahoma region of interest
region_name = "oklahoma"
plot_region_name = "Western Oklahoma"
# 34.00°-36.00°N, 98.00°-100.00°W
west, south, east, north = -100.0, 34.0, -98.0, 36.0

# The full 2024 calendar year, which spans the winter wheat harvest and the following
# autumn planting
date_range = [datetime(2024, 1, 1), datetime(2024, 12, 31)] # noqa: DTZ001
start, end = date_range

# Typical crop calendar for hard red winter wheat in Western Oklahoma, as the (month, day)
# each window starts and ends:
#   Harvest:  May 25 - June 30
#   Planting: September 1 - October 31 (early September for dual-purpose grazing
#             wheat, through October for grain-only wheat)
crop_calendar = [
    ((5, 25), (6, 30), "Wheat\nHarvest"),
    ((9, 1), (10, 31), "Wheat Planting /\nEmergence"),
]

years = sorted({str(dt.year) for dt in date_range})
year_str = "_".join(years)
plot_year_str = "-".join(years)


def format_coords(coord_left: float, coord_right: float, is_lat: bool = True) -> str:
    if is_lat:
        hem_left  = "N" if coord_left > 0.0 else "S"
        hem_right = "N" if coord_right > 0.0 else "S"
    else:
        hem_left  = "E" if coord_left > 0.0 else "W"
        hem_right = "E" if coord_right > 0.0 else "W"
    hem_left = "" if hem_right == hem_left else hem_left
    return f"{abs(coord_right)}°{hem_left}-{abs(coord_left)}°{hem_right}"


coords_str = f"{format_coords(north, south)}, {format_coords(west, east, is_lat=False)}"
plot_title = f"SIF-RCI over {plot_region_name} ({plot_year_str})\n{coords_str}"
plot2_title = f"SIF-RCI with Flash Drought Detections over {plot_region_name} ({plot_year_str})\n{coords_str}"


def doy_to_date(year, doy):
    return datetime(year, 1, 1) + timedelta(days=doy - 1) # noqa: DTZ001


doy_range: list[tuple[int, int]] = []
for year in range(start.year, end.year + 1):
    for doy in range(1, 365, 8):
        d = doy_to_date(year, doy)
        if start <= d <= end:
            doy_range.append((year, doy))

sif_time_series_fname = f"{region_name}_{year_str}_sif.csv"
swdi_time_series_fname = f"{region_name}_{year_str}_swdi.csv"
sif_csv_path = f"data/{sif_time_series_fname}"
swdi_csv_path = f"data/{swdi_time_series_fname}"

clim_dir = f"inputs/sif_increments_{region_name}"
fc_wp_path = f"inputs/{region_name}_fc_wp.tif"
raster_dir = f"data/{region_name}_time_steps"
fd_dir = f"data/{region_name}_fd"

# The hosted SMAP L4 virtual dataset references each granule by its file name at the
# time the dataset was built. NSIDC has since reprocessed part of the record (every
# granule from August 2023 on is now "Vv8012" rather than "Vv8010"/"Vv8011"), so those
# references fail with a 404 and the whole of 2024 is unreadable. patch_smap_references
# writes a local copy of the references that points at the current granules instead.
hosted_smap_refs = "https://its-live-data.s3-us-west-2.amazonaws.com/test-space/vds/SPL4SMGP.parquet"
smap_refs_path = "data/SPL4SMGP_refs_patched.json"

ts_fig_path = "inputs/WOklahoma_SIF_SWDI_2024.png"
detection_fig_path = "inputs/WOklahoma_FD_detection_2024.png"
map_fig_path = "inputs/WOklahoma_FD_maps_2024.png"
rci_gif_path = "inputs/WOklahoma_SIF_RCI_2024.gif"
zscore_gif_path = "inputs/WOklahoma_SIF_ZScore_2024.gif"
swdi_gif_path = "inputs/WOklahoma_SWDI_2024.gif"

# Thresholds shared by the detection rules and the plots
sif_rci_thresh = -0.5
moderate_swdi_thresh = -3
swdi_thresh = -5
area_thresh = 50

# Color scales shared by the maps and the animations
rci_norm = TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3)
zscore_norm = TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3)
swdi_norm = TwoSlopeNorm(vmin=-10, vcenter=0, vmax=10)

# Towns plotted on the maps for orientation, (longitude, latitude)
towns = {
    "Elk City": (-99.41, 35.41),
    "Weatherford": (-98.71, 35.53),
    "Hobart": (-99.09, 35.03),
    "Altus": (-99.33, 34.64),
    "Lawton": (-98.39, 34.61),
}

# The animations show the whole of Oklahoma (103.00°-94.43°W, 33.62°-37.00°N) around
# the region, so it can be placed within the rest of the western half of the state
state_west, state_south, state_east, state_north = -103.15, 33.4, -94.2, 37.25
state_towns = {
    "Elk City": towns["Elk City"],
    "Altus": towns["Altus"],
    "Lawton": towns["Lawton"],
    "Guymon": (-101.48, 36.69),
    "Woodward": (-99.40, 36.43),
    "Oklahoma City": (-97.52, 35.47),
    "Tulsa": (-95.93, 36.12),
}
state_labels = {
    "OKLAHOMA": (-96.3, 34.75),
    "TEXAS": (-96.9, 33.55),
    "KANSAS": (-98.0, 37.13),
}
# State polygons and boundaries from Natural Earth (1:10m), which build_state_boundaries
# saves for the area around the state the first time they're needed
natural_earth_url = "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_{}.zip"
states_path = f"inputs/{region_name}_states.geojson"


"""I. Inputs"""


def download_gosif() -> list[str]:
    output_dir = "data/gosif"
    os.makedirs(output_dir, exist_ok=True)

    gosif_geotiffs: list[str] = []
    for year, doy in doy_range:
        fname = download_unpack_gosif(year, day=doy, output_dir=output_dir, verbose=False)
        if fname:
            gosif_geotiffs.append(fname)
    return gosif_geotiffs


N_WINDOWS = 46  # GOSIF 8-day windows per year: DOY 1, 9 ... 361


def build_sif_climatology(start_year: int = 2001, end_year: int = 2020, ddof: int = 1) -> None:
    """Write the per-cell mean/std of SIF and its increment for each 8-day window.

    This is the climatology cell of 3_appendix.ipynb run over this region. One
    GeoTIFF is written per window j to `clim_dir` with bands (mean dSIF, std dSIF,
    mean SIF, std SIF), which is the layout compute_sif_time_series expects.
    """
    os.makedirs(clim_dir, exist_ok=True)
    gosif_geotiffs = sorted(glob.glob("data/gosif/GOSIF_???????.tif"))

    read_window = get_read_window(gosif_geotiffs[0], west, south, east, north)
    with rasterio.open(gosif_geotiffs[0]) as src:
        win_transform = src.window_transform(read_window)
        crs = src.crs

    # Step 1: accumulate increment grids per jth 8-day window
    increments_by_j: dict[int, list[np.ndarray]] = {j: [] for j in range(N_WINDOWS)}
    values_by_j: dict[int, list[np.ndarray]] = {j: [] for j in range(N_WINDOWS)}
    prev_grid: np.ndarray | None = None
    prev_ord: int | None = None

    for path in gosif_geotiffs:
        token = os.path.splitext(os.path.basename(path))[0].split("_")[1]
        year, doy = int(token[:4]), int(token[4:7])
        j = (doy - 1) // 8
        # Consecutive windows differ by 1, including across year boundaries
        ordi = year * N_WINDOWS + j

        # read 'start_year - 1' as lead-in (to seed the j=0 increment) but
        # store nothing from it. Ignore everything outside [start_year-1, end_year]
        if year < start_year - 1 or year > end_year:
            prev_grid, prev_ord = None, None
            continue

        grid = read_roi(path, read_window)
        if start_year <= year <= end_year:
            values_by_j[j].append(grid)
            # store only a genuinely consecutive increment assigned to the study period
            if prev_grid is not None and prev_ord == ordi - 1:
                increments_by_j[j].append(grid - prev_grid)

        prev_grid, prev_ord = grid, ordi

    # Step 2: per-pixel mean/std across years, write one GeoTIFF per j
    profile = {
        "driver": "GTiff",
        "height": int(read_window.height),
        "width": int(read_window.width),
        "count": 4,
        "dtype": "float32",
        "crs": crs,
        "transform": win_transform,
        "nodata": np.nan,
        "compress": "deflate",
    }
    for j in range(N_WINDOWS):
        if not values_by_j[j] or not increments_by_j[j]:
            print(f"window j={j:02d} has no data, skipped")
            continue

        istack = np.stack(increments_by_j[j], axis=0)
        vstack = np.stack(values_by_j[j], axis=0)

        # nanmean/nanstd warn on all-NaN pixels (water), which become NaN as intended
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_incr = np.nanmean(istack, axis=0)
            std_incr = np.nanstd(istack, axis=0, ddof=ddof)
            mean_sif = np.nanmean(vstack, axis=0)
            std_sif = np.nanstd(vstack, axis=0, ddof=ddof)

        doy_j = j * 8 + 1
        out_path = os.path.join(clim_dir, f"GOSIF_dSIF_clim_{doy_j:03d}.tif")
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(mean_incr.astype("float32"), 1)
            dst.write(std_incr.astype("float32"), 2)
            dst.write(mean_sif.astype("float32"), 3)
            dst.write(std_sif.astype("float32"), 4)
            dst.set_band_description(1, "mean dSIF")
            dst.set_band_description(2, "std dSIF")
            dst.set_band_description(3, "mean SIF")
            dst.set_band_description(4, "std SIF")
            dst.update_tags(
                window_index=str(j),
                doy=str(doy_j),
                start_year=str(start_year),
                end_year=str(end_year),
                ddof=str(ddof),
                units="W/m^2/sr/um",
            )


def compute_sif() -> None:
    if len(glob.glob(f"{clim_dir}/GOSIF_dSIF_clim_*.tif")) < N_WINDOWS:
        build_sif_climatology()
        print(f"SIF: Saved the climatology to {clim_dir}")

    if not os.path.exists(sif_csv_path):
        _, n_dates = compute_sif_time_series(
            download_gosif(),
            clim_dir,
            sif_time_series_fname,
            west, south, east, north,
            raster_dir=raster_dir,
        )
        print(f"SIF: Computed time series values for {n_dates} granules and saved to {sif_csv_path}")


def _granule_stamp(url: str) -> str:
    """SMAP_L4_SM_gph_20240902T043000_Vv8012_001.h5 -> 20240902T043000"""
    return os.path.basename(url).split("_")[4]


def _sm_rootzone_chunk(session, dmrpp_url: str) -> tuple[int, int] | None:
    """Read the (offset, size) of the single sm_rootzone chunk from a granule's DMR++."""
    response = session.get(dmrpp_url, timeout=120)
    if response.status_code != 200:
        return None
    var = re.search(r'<Float32 name="sm_rootzone">(.*?)</Float32>', response.text, re.S)
    if var is None:
        return None
    # The references assume one 1624 x 3856 chunk compressed with deflate level 3
    if 'deflateLevel="3"' not in var.group(1) or "1624 3856" not in var.group(1):
        return None
    chunk = re.findall(r'<dmrpp:chunk offset="(\d+)" nBytes="(\d+)"', var.group(1))
    if len(chunk) != 1:
        return None
    return int(chunk[0][0]), int(chunk[0][1])


def patch_smap_references(workers: int = 16) -> str:
    """Write the hosted SMAP L4 references to a local file, re-pointing stale granules.

    The current file name of every granule in the record is looked up in CMR. Where it
    differs from the referenced one, the sm_rootzone reference is replaced with the new
    file and the chunk offset and size from its DMR++ sidecar, the same source
    earthaccess.virtualize uses. The encoding of the reprocessed granules is unchanged,
    so the array metadata is kept as is. References that can't be patched are left
    alone, and are skipped as unreadable when the record is read.

    Returns:
        str: Path of the patched kerchunk JSON references.
    """
    if os.path.exists(smap_refs_path):
        return smap_refs_path

    import earthaccess
    import fsspec

    fs = fsspec.filesystem("reference", fo=hosted_smap_refs, remote_protocol="https")
    refs: dict[str, str | list] = {}
    for key in fs.references:
        # Keep only the soil moisture and its coordinates, the hosted references cover
        # every geophysical variable. The root group is opened unconsolidated, so its
        # consolidated metadata (which lists every variable) is dropped too.
        if key.split("/")[0] not in (".zattrs", ".zgroup", "sm_rootzone", "time", "x", "y"):
            continue
        value = fs.references[key]
        if isinstance(value, bytes):
            try:
                refs[key] = value.decode()
            except UnicodeDecodeError:
                refs[key] = "base64:" + base64.b64encode(value).decode()
        else:
            refs[key] = [value[0], *(int(v) for v in value[1:])]

    chunk_keys = [k for k in refs if k.startswith("sm_rootzone/") and isinstance(refs[k], list)]
    stamps = sorted(_granule_stamp(refs[k][0]) for k in chunk_keys)

    earthaccess.login()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        granules = earthaccess.search_data(
            short_name="SPL4SMGP",
            temporal=(datetime.strptime(stamps[0][:8], "%Y%m%d"), # noqa: DTZ007
                      datetime.strptime(stamps[-1][:8], "%Y%m%d") + timedelta(days=1)), # noqa: DTZ007
        )
    current = {
        _granule_stamp(link): link
        for granule in granules
        for link in granule.data_links()
        if link.endswith(".h5")
    }

    stale = [k for k in chunk_keys if current.get(_granule_stamp(refs[k][0]), refs[k][0]) != refs[k][0]]
    print(f"SMAP: Re-pointing {len(stale)} of {len(chunk_keys)} references at reprocessed granules")

    session = earthaccess.get_requests_https_session()
    with ThreadPoolExecutor(workers) as pool:
        new_urls = [current[_granule_stamp(refs[k][0])] for k in stale]
        chunks = list(pool.map(lambda url: _sm_rootzone_chunk(session, url + ".dmrpp"), new_urls))
    n_failed = 0
    for key, url, chunk in zip(stale, new_urls, chunks):
        if chunk is None:
            n_failed += 1
            continue
        refs[key] = [url, *chunk]
    if n_failed:
        print(f"SMAP: {n_failed} references could not be patched and will be skipped")

    with open(smap_refs_path, "w") as f:
        json.dump({"version": 1, "refs": refs}, f)
    return smap_refs_path


def compute_swdi() -> None:
    if not os.path.exists(swdi_csv_path):
        compute_swdi_timeseries(
            start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d"),
            swdi_time_series_fname,
            (west, south, east, north),
            ref_url=patch_smap_references(),
            raster_dir=raster_dir,
            fc_wp_path=fc_wp_path,
        )
        print(f"SWDI: Saved time series values to {swdi_csv_path}")


def detect() -> None:
    detect_flash_drought_sif(raster_dir, fd_dir, sif_csv_path, threshold=sif_rci_thresh)
    detect_flash_drought_swdi(raster_dir, fd_dir, swdi_csv_path, abs_threshold=swdi_thresh)


def build_state_boundaries() -> None:
    """Save the state polygons and the boundaries between states around Oklahoma.

    The Natural Earth 1:10m states (with the larger lakes cut out as holes) and the
    lines between them are read straight from their zip archives, clipped to a margin
    around the extent of the animations and written to `states_path` as GeoJSON, so
    the plots only need the json module to draw them.
    """
    if os.path.exists(states_path):
        return

    import pyogrio.raw  # type: ignore
    import shapely  # type: ignore

    clip_box = (state_west - 1.5, state_south - 1.5, state_east + 1.5, state_north + 1.5)
    features = []
    for layer, kind, columns in [
        ("admin_1_states_provinces_lakes", "state", ["postal"]),
        ("admin_1_states_provinces_lines", "boundary", []),
    ]:
        _, _, geometry, fields = pyogrio.raw.read(natural_earth_url.format(layer), bbox=clip_box, columns=columns)
        postals = fields[0] if columns else [None] * len(geometry)
        for postal, geom in zip(postals, shapely.from_wkb(geometry)):
            geom = shapely.set_precision(shapely.clip_by_rect(geom, *clip_box), 1e-4)
            if geom.is_empty:
                continue
            # Counter-clockwise exteriors and clockwise holes, so matplotlib fills them
            geom = shapely.orient_polygons(geom)
            features.append({
                "type": "Feature",
                "properties": {"kind": kind, "postal": postal},
                "geometry": json.loads(shapely.to_geojson(geom)),
            })

    with open(states_path, "w") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)
    print(f"Saved the state boundaries to {states_path}")


"""II. Time series plots"""


class MonthYearFormatter(mdates.DateFormatter):
    """Overengineered x axis tick label formatter."""

    def format_ticks(self, values):
        dates = [mdates.num2date(v) for v in values]
        labels = []
        last_year = None
        for d in dates:
            if d.year != last_year:
                labels.append(d.strftime("%b %Y"))
                last_year = d.year
            else:
                labels.append(d.strftime("%b"))
        return labels


def read_time_series(csv_path: str, date_col: str, cols: list[str]) -> tuple[list[datetime], dict[str, list[float]]]:
    """Read the date column and the requested value columns of a time series CSV."""
    dates: list[datetime] = []
    values: dict[str, list[float]] = {col: [] for col in cols}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dates.append(datetime.strptime(row[date_col], "%Y-%m-%d")) # noqa: DTZ007
            for col in cols:
                values[col].append(float(row[col]) if row.get(col) else np.nan)
    return dates, values


def add_crop_calendar(ax: plt.Axes, label: bool = True, year: int = start.year) -> None:
    """Cross-hatch the year's usual harvest and planting windows and label them on the plot."""
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for (start_month, start_day), (end_month, end_day), text in crop_calendar:
        window_start = datetime(year, start_month, start_day) # noqa: DTZ001
        window_end = datetime(year, end_month, end_day) # noqa: DTZ001
        ax.axvspan(window_start, window_end, facecolor="none", edgecolor="grey", # type: ignore
                   hatch="///", linewidth=0.0, alpha=0.2, zorder=0)
        if label:
            # x in data coordinates, y in axes-fraction coordinates
            ax.text(window_start + (window_end - window_start) / 2, 0.98, # type: ignore
                    text, transform=trans, ha="center", va="top", fontsize=10, color="black")


def leave_room_for_crop_labels(*axes: plt.Axes, label_height: float = 0.15) -> None:
    """Raise the top of each y-axis so the data stays below the crop calendar labels.

    The labels take up about label_height of the axes height, at the top.
    """
    for ax in axes:
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom, bottom + (top - bottom) / (1 - label_height))


def plot_swdi_series(
    ax: plt.Axes, dates: list[datetime], swdi: list[float], title: str, label_crops: bool = False,
) -> None:
    """Plot the SWDI against the deficit thresholds, shading where it's below the severe deficit."""
    add_crop_calendar(ax, label=label_crops, year=dates[0].year)
    flash_drought_swdi = [r < swdi_thresh for r in swdi]

    ax.set_ylim(min(-11, np.nanmin(swdi) - 1), max(0, np.nanmax(swdi) + 1))
    if label_crops:
        leave_room_for_crop_labels(ax)
    ax.axhline(moderate_swdi_thresh, color="goldenrod", linestyle=":", linewidth=1, label="Moderate deficit")
    ax.axhline(swdi_thresh, color="red", linestyle=":", linewidth=1, label="Severe deficit")
    ax.plot(dates, swdi, color="darkblue", linewidth=1.5, label=f"{dates[0].year} SWDI") # type: ignore
    ax.fill_between(dates, swdi, swdi_thresh, where=flash_drought_swdi, color="brown", alpha=0.1) # type: ignore
    ax.set_ylabel("SWDI")
    ax.set_title(title)
    ax.legend(loc="lower left")


def plot_time_series() -> None:
    """The SIF Z-Score, SIF-RCI and SWDI figure from section I of 2_detection.ipynb."""
    plot_dates, sif = read_time_series(sif_csv_path, "date", ["sif_zscore", "sif_rci"])
    plot_zscore, plot_sifrci = sif["sif_zscore"], sif["sif_rci"]

    plot2_dates, swdi = read_time_series(swdi_csv_path, "time", ["swdi"])
    # Our time series of SWDI is slightly longer, so skip dates that aren't
    # in the SIF time series
    keep = [plot_dates[0] <= d <= plot_dates[-1] for d in plot2_dates]
    plot2_dates = [d for d, k in zip(plot2_dates, keep) if k]
    plot2_swdi = [s for s, k in zip(swdi["swdi"], keep) if k]

    # Highlight the periods where the SIF-RCI signal was below 0
    flash_drought = [r < -0.05 for r in plot_sifrci]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    add_crop_calendar(ax1)

    ax1.axhline(0, color="black", linestyle=":", linewidth=1)
    ax1.plot(plot_dates, plot_zscore, color="darkgreen", linewidth=0.5, label="SIF Z-Score")
    ax1.plot(plot_dates, plot_sifrci, marker="o", color=(0.77, 0.64, 0.52), linewidth=2, markersize=4, label="SIF-RCI")
    ax1.fill_between(plot_dates, plot_sifrci, 0, where=flash_drought, color="brown", alpha=0.1)
    ax1.set_title(plot_title)
    ax1.legend(loc="lower left")

    plot_swdi_series(ax2, plot2_dates, plot2_swdi, f"SWDI ({plot_year_str})")

    ax2.xaxis.set_major_formatter(MonthYearFormatter("%b %Y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(ts_fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved {ts_fig_path}")


def find_streaks(flags: list[bool]) -> list[tuple[int, int]]:
    """Find each contiguous run of True values in flags as (start_index, end_index) pairs."""
    streaks: list[tuple[int, int]] = []
    streak_start: int | None = None
    for i, flag in enumerate(flags):
        if flag and streak_start is None:
            streak_start = i
        elif not flag and streak_start is not None:
            streaks.append((streak_start, i - 1))
            streak_start = None
    if streak_start is not None:
        streaks.append((streak_start, len(flags) - 1))
    return streaks


def add_flash_drought_markers(ax: plt.Axes, dates: list[datetime], flags: list[bool]) -> None:
    """Draw vertical dashed lines at the start and end of each contiguous flash drought streak."""
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)

    for streak_num, (start_idx, end_idx) in enumerate(find_streaks(flags)):
        first_date, last_date = dates[start_idx], dates[end_idx]
        # Shade each streak span (only label the first one so the legend has a single entry)
        ax.axvspan(first_date, last_date, color="brown", alpha=0.1, # type: ignore
                   label="Flash Drought" if streak_num == 0 else None)

        # Draw a labelled vertical line at the start and end of the streak. A
        # single-time-step streak has a coincident start/end, so only mark it once.
        # Start dates are labelled at the top and end dates at the bottom so the end
        # of one streak and the start of the next don't overlap when they are close.
        marks = [(first_date, "right", 0.98, "top")]
        if last_date != first_date:
            marks.append((last_date, "left", 0.02, "bottom"))
        for d, ha, y, va in marks:
            if d == dates[-1]:
                continue
            ax.axvline(d, color="brown", linestyle="--", linewidth=1, zorder=1) # type: ignore
            ax.text(d, y, d.strftime("%Y-%m-%d"), transform=trans, ha=ha, va=va, # type: ignore
                    fontsize=9, color="brown", rotation=90,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1})


def plot_swdi_detection(
    ax: plt.Axes, dates: list[datetime], swdi: list[float], fd_percent: list[float], title: str,
    label_crops: bool = False,
) -> None:
    """Plot the SWDI and the percent of the region's grid cells in flash drought on a twin axis.

    The periods with more than area_thresh percent of the cells in flash drought are
    marked as a flash drought over the whole region.
    """
    flash_drought_swdi = [r > area_thresh for r in fd_percent]
    swdi_color = "darkblue"
    percent_color = "slategray"

    add_crop_calendar(ax, label=label_crops, year=dates[0].year)
    ax.plot(dates, swdi, marker="o", color=swdi_color, linewidth=2, markersize=4, label="SWDI") # type: ignore
    ax.axhline(swdi_thresh, color="black", linestyle=":", linewidth=1)
    ax.tick_params(axis="y", labelcolor=swdi_color)
    ax.set_ylim(min(-12, np.nanmin(swdi) - 1), max(2, np.nanmax(swdi) + 1))
    ax.set_ylabel("SWDI")
    ax.set_title(title)

    percent_ax = ax.twinx()
    percent_ax.plot(dates, fd_percent, marker="^", color=percent_color, linewidth=1, markersize=4, # type: ignore
                    label="% Area in Drought")
    percent_ax.tick_params(axis="y", labelcolor=percent_color)
    percent_ax.set_ylim(0, 100)
    if label_crops:
        leave_room_for_crop_labels(ax, percent_ax)
    percent_ax.set_ylabel("Percent (%)")
    add_flash_drought_markers(percent_ax, dates, flash_drought_swdi)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = percent_ax.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="lower left")


def plot_detection() -> None:
    """The flash drought detection figure from section II of 2_detection.ipynb."""
    plot_dates, sif = read_time_series(sif_csv_path, "date", ["sif_rci", "fd_percent"])
    plot_sifrci, plot_fdperc = sif["sif_rci"], sif["fd_percent"]
    plot2_dates, swdi = read_time_series(swdi_csv_path, "time", ["swdi", "fd_percent"])
    plot2_swdi, plot2_fdperc = swdi["swdi"], swdi["fd_percent"]

    flash_drought_sif = [r > area_thresh for r in plot_fdperc]

    # Colors for top subplot
    color1 = (0.77, 0.64, 0.52)
    color2 = "darkgreen"

    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    """Subplot 1: SIF Flash Drought Detection"""
    add_crop_calendar(ax1)
    ax1.plot(plot_dates, plot_sifrci, marker="o", color=color1, linewidth=2, markersize=4, label="SIF-RCI")
    ax1.axhline(0, color="black", linestyle=":", linewidth=1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ylim = max([abs(r) for r in plot_sifrci])
    ax1.set_ylim(-1.05 * ylim, 1.05 * ylim)
    ax1.set_ylabel("RCI")
    ax1.set_title(plot2_title)

    ax2 = ax1.twinx()
    ax2.plot(plot_dates, plot_fdperc, marker="^", color=color2, linewidth=1, markersize=4, label="% Area in Drought")
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("Percent (%)")
    add_flash_drought_markers(ax2, plot_dates, flash_drought_sif)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left")

    """Subplot 2: SWDI Flash Drought Detection"""
    plot_swdi_detection(ax3, plot2_dates, plot2_swdi, plot2_fdperc, f"SWDI ({plot_year_str})")

    ax3.xaxis.set_major_formatter(MonthYearFormatter("%b %Y"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator())
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(detection_fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved {detection_fig_path}")


"""III. Maps of the drought's development"""


def date_stamp(d: datetime) -> str:
    """The date as it appears in the names of the per-step rasters."""
    return f"{d.year}_{d.month:02d}_{d.day:02d}"


def read_lonlat_grid(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read band 1 of a raster with the longitude and latitude of its cell edges.

    The SIF rasters are already on a lon/lat grid. The SWDI rasters are on the
    EASE-Grid 2.0 Global projection, which is cylindrical so x depends only on
    longitude and y only on latitude, and the cell edges convert independently.
    """
    with rasterio.open(path) as src:
        grid = src.read(1).astype("float64")
        x_edges = src.transform.c + src.transform.a * np.arange(src.width + 1)
        y_edges = src.transform.f + src.transform.e * np.arange(src.height + 1)
        crs = src.crs
    if crs.to_epsg() == 4326:
        return x_edges, y_edges, grid
    transformer = Transformer.from_crs(EASE2_GLOBAL_EPSG, "EPSG:4326", always_xy=True)
    lon_edges, _ = transformer.transform(x_edges, np.zeros_like(x_edges))
    _, lat_edges = transformer.transform(np.zeros_like(y_edges), y_edges)
    return np.asarray(lon_edges), np.asarray(lat_edges), grid


def detection_hatch(edgecolor: str, hatch_linewidth: float, cells=(), hatch: str = "///") -> PatchCollection:
    """A hatched, unfilled collection, drawn twice to give the hatch lines a halo."""
    collection = PatchCollection(list(cells), facecolor="none", edgecolor=edgecolor, linewidth=0.0, hatch=hatch)
    collection.set_hatch_linewidth(hatch_linewidth)
    return collection


def hatch_detections(
    ax: plt.Axes, lon_edges: np.ndarray, lat_edges: np.ndarray, detection: np.ndarray, hatch: str = "///",
) -> list[Collection]:
    """Hatch the grid cells flagged as flash drought, returning the hatch collections.

    The hatching is black over a wider white line so it stays visible over both
    the pale and the darkest cells of the colormap.
    """
    rows, cols = np.nonzero(detection > 0.5)
    cells = [
        Rectangle((lon_edges[c], lat_edges[r + 1]), lon_edges[c + 1] - lon_edges[c], lat_edges[r] - lat_edges[r + 1])
        for r, c in zip(rows, cols)
    ]
    return [ax.add_collection(detection_hatch("white", 2.5, cells, hatch)),
            ax.add_collection(detection_hatch("black", 0.8, cells, hatch))]


@functools.cache
def read_state_boundaries() -> tuple[dict[str, list[list[np.ndarray]]], list[np.ndarray]]:
    """Read the rings of each state's polygons by postal code, and the boundary lines."""
    build_state_boundaries()
    with open(states_path) as f:
        features = json.load(f)["features"]

    states: dict[str, list[list[np.ndarray]]] = {}
    lines: list[np.ndarray] = []
    for feature in features:
        geometry = feature["geometry"]
        parts = geometry["coordinates"] if geometry["type"].startswith("Multi") else [geometry["coordinates"]]
        if feature["properties"]["kind"] == "state":
            polygons = states.setdefault(feature["properties"]["postal"], [])
            polygons.extend([np.asarray(ring) for ring in polygon] for polygon in parts)
        else:
            lines.extend(np.asarray(line) for line in parts)
    return states, lines


def add_map_context(ax: plt.Axes, label_towns: bool) -> None:
    """Draw the state boundaries and town markers, and set the map extent."""
    _, lines = read_state_boundaries()
    for line in lines:
        ax.plot(line[:, 0], line[:, 1], color="black", linewidth=1)
    for name, (lon, lat) in towns.items():
        ax.plot(lon, lat, marker="o", markersize=3, color="black")
        if label_towns:
            ax.text(lon, lat + 0.04, name, fontsize=7, ha="center", va="bottom",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.6, "pad": 0.5})
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_aspect(1 / np.cos(np.deg2rad((south + north) / 2)))
    ax.set_xticks([])
    ax.set_yticks([])


def sif_zscore_grid(sif_date: datetime) -> np.ndarray:
    """Per-cell SIF Z-Score, which compute_sif_time_series only saves as a spatial average."""
    doy = sif_date.timetuple().tm_yday
    geotiff = f"data/gosif/GOSIF_{sif_date.year}{doy:03d}.tif"
    sif_grid = read_roi(geotiff, get_read_window(geotiff, west, south, east, north))
    with rasterio.open(f"{clim_dir}/GOSIF_dSIF_clim_{doy:03d}.tif") as clim_src:
        mean_sif_band = clim_src.read(3).astype(float)
        std_sif_band = clim_src.read(4).astype(float)
    return (sif_grid - mean_sif_band) / std_sif_band


def draw_swdi_map(
    ax: plt.Axes, swdi_date: datetime, raster_dir: str, fd_dir: str, label_towns: bool = False,
) -> QuadMesh:
    """Map the SWDI per grid cell at a time step, hatching the cells with a flash drought detection."""
    stamp = date_stamp(swdi_date)
    lon_edges, lat_edges, swdi_grid = read_lonlat_grid(f"{raster_dir}/swdi_{stamp}.tif")
    mesh = ax.pcolormesh(lon_edges, lat_edges, swdi_grid, cmap="BrBG", norm=swdi_norm)
    _, _, detection = read_lonlat_grid(f"{fd_dir}/fd_swdi_{stamp}.tif")
    hatch_detections(ax, lon_edges, lat_edges, detection)
    add_map_context(ax, label_towns)
    return mesh


def plot_maps(map_dates: list[tuple[datetime, str]]) -> None:
    """Small multiples of SIF-RCI, SIF Z-Score and SWDI per grid cell, with detections hatched.

    Arguments:
        map_dates: (date, label) for each column. SIF has an 8-day cadence and SWDI a
            3-day one, so each row shows its time step closest to the date.
    """
    sif_dates, _ = read_time_series(sif_csv_path, "date", [])
    swdi_dates, _ = read_time_series(swdi_csv_path, "time", [])

    n_cols = len(map_dates)
    fig, axes = plt.subplots(3, n_cols, figsize=(2.15 * n_cols + 1.0, 9.0), squeeze=False, layout="constrained")

    for col, (map_date, label) in enumerate(map_dates):
        sif_date = min(sif_dates, key=lambda d: abs(d - map_date))
        swdi_date = min(swdi_dates, key=lambda d: abs(d - map_date))
        sif_stamp = date_stamp(sif_date)

        # Row 1: SIF-RCI, the early warning of slowed vegetation growth
        ax = axes[0, col]
        lon_edges, lat_edges, rci_grid = read_lonlat_grid(f"{raster_dir}/sif_rci_{sif_stamp}.tif")
        rci_mesh = ax.pcolormesh(lon_edges, lat_edges, rci_grid, cmap="BrBG", norm=rci_norm)
        _, _, detection = read_lonlat_grid(f"{fd_dir}/fd_sifrci_{sif_stamp}.tif")
        hatch_detections(ax, lon_edges, lat_edges, detection)
        add_map_context(ax, label_towns=col == 0)
        ax.set_title(f"{label}\n{sif_date:%b %d}", fontsize=11)

        # Row 2: SIF Z-Score, the anomaly in vegetation productivity itself
        ax = axes[1, col]
        zscore_mesh = ax.pcolormesh(lon_edges, lat_edges, sif_zscore_grid(sif_date), cmap="BrBG", norm=zscore_norm)
        add_map_context(ax, label_towns=False)
        ax.set_title(f"{sif_date:%b %d}", fontsize=10)

        # Row 3: SWDI, the root zone soil water deficit
        ax = axes[2, col]
        swdi_mesh = draw_swdi_map(ax, swdi_date, raster_dir, fd_dir)
        ax.set_title(f"{swdi_date:%b %d}", fontsize=10)

    for row, (mesh, name) in enumerate([(rci_mesh, "SIF-RCI"), (zscore_mesh, "SIF Z-Score"), (swdi_mesh, "SWDI")]):
        axes[row, 0].set_ylabel(name, fontsize=12)
        fig.colorbar(mesh, ax=axes[row, :].tolist(), label=name, extend="both", shrink=0.85)

    hatch_handle = Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="black", linewidth=0.0, hatch="///")
    fig.legend(
        [hatch_handle],
        [(f"Flash drought detected in grid cell (SIF-RCI < {sif_rci_thresh} for 3 steps, or SWDI "
          f"fell by 2+ over 30 days to {swdi_thresh} or below)")],
        loc="outside lower center", frameon=False, fontsize=10,
    )
    fig.suptitle(f"Development of the {plot_year_str} Flash Drought over {plot_region_name}\n{coords_str}", fontsize=13)
    fig.savefig(map_fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {map_fig_path}")


"""IV. Animations of each index"""

# Each frame is shown for a time proportional to its time step, so the year takes as
# long to play for the 8-day SIF as for the 3-day SWDI
ms_per_day = 50
final_frame_ms = 2500

primary_ink = "#1a1a1a"
secondary_ink = "#555555"
threshold_color = "red"
# The grid cells are only a few pixels across in the animations, so the hatching of
# the detections is sparser than on the maps to leave more of the index visible
animation_hatch = "//"


def signed(value: float, fmt: str = ".2f") -> str:
    """Format a number with a minus sign rather than a hyphen, as on the plot axes."""
    return format(value, fmt).replace("-", "\N{MINUS SIGN}")


def add_state_context(ax: plt.Axes) -> None:
    """Shade Oklahoma and its neighbors, outline the region and label the towns, and set the extent."""
    states, lines = read_state_boundaries()
    # The larger lakes are holes in the state polygons, so they show the background
    ax.set_facecolor("#d3e2ec")
    for postal, polygons in states.items():
        for rings in polygons:
            path = Path.make_compound_path(*(Path(ring) for ring in rings))
            ax.add_patch(PathPatch(path, facecolor="#e8e4dc" if postal == "OK" else "#f8f7f4",
                                   edgecolor="none", zorder=0))
    for line in lines:
        ax.plot(line[:, 0], line[:, 1], color="#3a3a3a", linewidth=0.9, zorder=2)
    ax.add_patch(Rectangle((west, south), east - west, north - south, facecolor="none",
                           edgecolor="black", linewidth=1.5, zorder=3))

    halo = [patheffects.withStroke(linewidth=2.5, foreground="white")]
    for name, (lon, lat) in state_towns.items():
        ax.plot(lon, lat, marker="o", markersize=4, color="black", markeredgecolor="white",
                markeredgewidth=0.6, zorder=4)
        # Towns in the region are labelled above their marker, and the rest to the right
        if west <= lon <= east and south <= lat <= north:
            ax.text(lon, lat + 0.05, name, fontsize=8, ha="center", va="bottom", color=primary_ink,
                    path_effects=halo, zorder=4)
        else:
            ax.text(lon + 0.07, lat, name, fontsize=8, ha="left", va="center", color=primary_ink,
                    path_effects=halo, zorder=4)
    for name, (lon, lat) in state_labels.items():
        ax.text(lon, lat, name, fontsize=11, ha="center", va="center", color="#8c8780", zorder=1)

    ax.set_xlim(state_west, state_east)
    ax.set_ylim(state_south, state_north)
    ax.set_aspect(1 / np.cos(np.deg2rad((state_south + state_north) / 2)))
    ax.set_xticks([])
    ax.set_yticks([])


def save_gif(gif_path: str, frames: list[np.ndarray], durations: list[int], reserved_colors: np.ndarray) -> None:
    """Save RGB frames as a looping GIF.

    Every frame is mapped to one 256-color palette, so the parts of the figure that
    don't change don't flicker. The palette holds the reserved colors (RGBA bytes),
    such as each of the colormap's colors so the index is shown without banding, and
    the rest of the figure's colors as taken from a sample of the frames.
    """
    n_other = 256 - len(reserved_colors)
    sample = np.concatenate(frames[:: max(1, len(frames) // 8)], axis=0)
    other_colors = Image.fromarray(sample).quantize(colors=n_other, method=Image.Quantize.MEDIANCUT).getpalette() or []
    palette = Image.new("P", (1, 1))
    palette.putpalette(reserved_colors[:, :3].ravel().tolist() + other_colors[: 3 * n_other])
    images = [Image.fromarray(frame).quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=durations, loop=0)
    print(f"Saved {gif_path}")


def animate_map(
    gif_path: str,
    name: str,
    description: str,
    dates: list[datetime],
    grids: list[np.ndarray],
    lon_edges: np.ndarray,
    lat_edges: np.ndarray,
    norm: TwoSlopeNorm,
    series: list[float],
    threshold: tuple[float, str] | None = None,
    detections: list[np.ndarray] | None = None,
    fd_percent: list[float] | None = None,
    detection_rule: str = "",
) -> None:
    """Animate an index per grid cell over the state, with its regional mean on a timeline.

    The timeline spans the whole calendar year of the first date, so a year that's still
    under way is shown as such.

    Arguments:
        gif_path: Path of the GIF to write.
        name: Name of the index, for the titles and colorbar.
        description: One line describing the index, shown under the title.
        dates: Date of each time step.
        grids: Index per grid cell at each time step, on the grid given by the edges.
        lon_edges, lat_edges: Longitude and latitude of the grid cell edges.
        norm: Color scale of the index.
        series: Regional mean of the index at each time step.
        threshold: (value, label) of a threshold to mark on the timeline.
        detections: Flash drought detection per grid cell at each time step, hatched.
        fd_percent: Percent of the grid cells with a detection at each time step.
        detection_rule: Short statement of the detection rule, for the legend.
    """
    # About half of the 256 colors a GIF can hold are left for the colormap. An odd number
    # of colors puts the neutral color at its center, where the index is 0.
    cmap = plt.get_cmap("BrBG").resampled(127)
    year = dates[0].year
    year_start, year_end = datetime(year, 1, 1), datetime(year, 12, 31) # noqa: DTZ001
    fig = plt.figure(figsize=(9.8, 7.2), dpi=100, layout="constrained")
    grid_spec = fig.add_gridspec(2, 2, height_ratios=[4.6, 1.5], width_ratios=[1, 0.022])
    ax = fig.add_subplot(grid_spec[0, 0])
    cax = fig.add_subplot(grid_spec[0, 1])
    ts_ax = fig.add_subplot(grid_spec[1, 0])
    fig.suptitle(f"{name} over {plot_region_name}, {year}", fontsize=15, fontweight="bold")
    ax.set_title(description, fontsize=10, color=secondary_ink)

    # The map. The date and regional statistics go over Texas, south of the panhandle.
    add_state_context(ax)
    mesh = ax.pcolormesh(lon_edges, lat_edges, grids[0], cmap=cmap, norm=norm, zorder=1)
    fig.colorbar(mesh, cax=cax, label=name, extend="both")
    date_text = ax.text(0.055, 0.77, "", transform=ax.transAxes, fontsize=22, fontweight="bold",
                        va="top", color=primary_ink)
    stats_text = ax.text(0.055, 0.665, "", transform=ax.transAxes, fontsize=11, va="top",
                         color=primary_ink, linespacing=1.6)
    handles = [Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="black", linewidth=1.5)]
    labels = [f"Study region\n({coords_str})"]
    if detections is not None:
        handles.append(Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="black", linewidth=0.0,
                                 hatch=animation_hatch))
        labels.append(f"Flash drought detected\n({detection_rule})")
    ax.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.045, 0.02), frameon=False, fontsize=9,
              handleheight=1.6)

    # The timeline of the regional mean, drawn up to the current time step over the whole year in grey
    lo = min(np.nanmin(series), threshold[0] if threshold else 0, 0)
    hi = max(np.nanmax(series), 0)
    pad = 0.12 * (hi - lo)
    ts_ax.axhline(0, color="black", linestyle=":", linewidth=0.8)
    if threshold is not None:
        # Not antialiased, as pink blends of the line with the background would be mapped
        # to the colormap's tans in the GIF palette
        ts_ax.axhline(threshold[0], color=threshold_color, linestyle=":", linewidth=1, antialiased=False)
        # Labelled at the end of the year, on the other side of the line from where the
        # series ends so the last time steps don't cover it
        ends_above = series[-1] > threshold[0]
        ts_ax.text(year_end, threshold[0], f"{threshold[1]} ({signed(threshold[0], 'g')})", ha="right", # type: ignore
                   va="top" if ends_above else "bottom", fontsize=8, color=secondary_ink)
    ts_ax.plot(dates, series, color="#cccccc", linewidth=1.5) # type: ignore
    past_line, = ts_ax.plot([], [], color="#2b2b2b", linewidth=1.75)
    now_line = ts_ax.axvline(dates[0], color="#999999", linewidth=0.8) # type: ignore
    now_marker, = ts_ax.plot([], [], marker="o", markersize=9, markeredgecolor="black", markeredgewidth=0.8,
                             linestyle="none", zorder=3)
    ts_ax.set_xlim(year_start, year_end) # type: ignore
    ts_ax.set_ylim(lo - pad, hi + pad)
    ts_ax.xaxis.set_major_locator(mdates.MonthLocator())
    ts_ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ts_ax.set_title(f"Regional mean {name}", loc="left", fontsize=10)
    ts_ax.spines[["top", "right"]].set_visible(False)

    frames: list[np.ndarray] = []
    hatches: list[Collection] = []
    for i, (date, grid) in enumerate(zip(dates, grids)):
        mesh.set_array(grid)
        for collection in hatches:
            collection.remove()
        if detections is not None:
            hatches = hatch_detections(ax, lon_edges, lat_edges, detections[i], animation_hatch)

        date_text.set_text(f"{date:%b} {date.day}, {date.year}")
        stats = [f"Regional mean {signed(series[i])}"]
        if fd_percent is not None:
            stats.append(f"Flash drought in {fd_percent[i]:.0f}% of cells")
        stats_text.set_text("\n".join(stats))

        past_line.set_data(dates[: i + 1], series[: i + 1]) # type: ignore
        now_line.set_xdata([date, date]) # type: ignore
        now_marker.set_data([date], [series[i]]) # type: ignore
        now_marker.set_markerfacecolor(cmap(norm(series[i])))

        fig.canvas.draw()
        if i == 0:
            # Fix the layout from the first frame, so nothing moves between frames
            fig.set_layout_engine("none")
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()) # type: ignore
    plt.close(fig)

    durations = [(dates[1] - dates[0]).days * ms_per_day] * len(frames)
    durations[-1] += final_frame_ms
    # The threshold line is too thin to be given its color from the sample of the frames
    threshold_rgba = np.round(np.asarray(to_rgba(threshold_color)) * 255).astype(np.uint8)
    reserved_colors = np.vstack([cmap(np.arange(cmap.N), bytes=True), threshold_rgba])
    save_gif(gif_path, frames, durations, reserved_colors)


def animate_swdi(gif_path: str, csv_path: str, raster_dir: str, fd_dir: str) -> None:
    """Animated GIF of the SWDI per grid cell through the year.

    Arguments:
        gif_path: Path of the GIF to write.
        csv_path: Path of the SWDI time series CSV, with its fd_percent column.
        raster_dir: Directory of the per-step SWDI rasters.
        fd_dir: Directory of the per-step flash drought detection rasters.
    """
    dates, swdi = read_time_series(csv_path, "time", ["swdi", "fd_percent"])
    rasters = [read_lonlat_grid(f"{raster_dir}/swdi_{date_stamp(d)}.tif") for d in dates]
    lon_edges, lat_edges, _ = rasters[0]
    animate_map(
        gif_path,
        "SWDI",
        ("Root zone Soil Water Deficit Index from SMAP L4, as 3-day means. "
         f"0 is field capacity and {signed(-10, 'g')} the wilting point."),
        dates,
        [grid for _, _, grid in rasters],
        lon_edges, lat_edges,
        swdi_norm,
        swdi["swdi"],
        threshold=(swdi_thresh, "Severe deficit"),
        detections=[read_lonlat_grid(f"{fd_dir}/fd_swdi_{date_stamp(d)}.tif")[2] for d in dates],
        fd_percent=swdi["fd_percent"],
        detection_rule=f"SWDI fell 2+ in 30 days to {signed(swdi_thresh, 'g')} or below",
    )


def animate_indices() -> None:
    """Animated GIFs of the SIF-RCI, SIF Z-Score and SWDI per grid cell through the year."""
    sif_dates, sif = read_time_series(sif_csv_path, "date", ["sif_zscore", "sif_rci", "fd_percent"])

    rci_rasters = [read_lonlat_grid(f"{raster_dir}/sif_rci_{date_stamp(d)}.tif") for d in sif_dates]
    sif_lon_edges, sif_lat_edges, _ = rci_rasters[0]
    animate_map(
        rci_gif_path,
        "SIF-RCI",
        ("SIF Rapid Change Index from GOSIF, in 8-day steps. "
         "Below 0, SIF has been rising more slowly or falling faster than usual."),
        sif_dates,
        [grid for _, _, grid in rci_rasters],
        sif_lon_edges, sif_lat_edges,
        rci_norm,
        sif["sif_rci"],
        threshold=(sif_rci_thresh, "Detection threshold"),
        detections=[read_lonlat_grid(f"{fd_dir}/fd_sifrci_{date_stamp(d)}.tif")[2] for d in sif_dates],
        fd_percent=sif["fd_percent"],
        detection_rule=f"SIF-RCI < {signed(sif_rci_thresh, 'g')} for 3 steps",
    )

    animate_map(
        zscore_gif_path,
        "SIF Z-Score",
        ("Anomaly of GOSIF SIF from its 2001-2020 climatology, in 8-day steps. "
         "Below 0, vegetation is less productive than usual."),
        sif_dates,
        [sif_zscore_grid(d) for d in sif_dates],
        sif_lon_edges, sif_lat_edges,
        zscore_norm,
        sif["sif_zscore"],
    )

    animate_swdi(swdi_gif_path, swdi_csv_path, raster_dir, fd_dir)


def main() -> None:
    compute_sif()
    compute_swdi()
    detect()
    plot_time_series()
    plot_detection()
    # The phases of the event, following the time series of SWDI and SIF-RCI
    plot_maps([
        (datetime(2024, 6, 1), "Before onset"), # noqa: DTZ001
        (datetime(2024, 6, 25), "Onset"), # noqa: DTZ001
        (datetime(2024, 8, 4), "Intensification"), # noqa: DTZ001
        (datetime(2024, 8, 20), "Peak SIF-RCI"), # noqa: DTZ001
        (datetime(2024, 10, 15), "Peak severity"), # noqa: DTZ001
        (datetime(2024, 11, 8), "Recovery"), # noqa: DTZ001
    ])
    animate_indices()


if __name__ == "__main__":
    with contextlib.chdir(os.path.dirname(os.path.abspath(__file__))):
        main()
