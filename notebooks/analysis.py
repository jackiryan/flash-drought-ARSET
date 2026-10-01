import csv
import glob
import logging
import os
import time
import warnings
from datetime import datetime
from typing import TypeVar

import earthaccess
import fsspec
import numpy as np
import rasterio
import xarray as xr
import zarr
from dask.distributed import Client, LocalCluster
from pyproj import Transformer
from rasterio.transform import from_origin
from rasterio.windows import Window, from_bounds

XarrayObj = TypeVar("XarrayObj", xr.Dataset, xr.DataArray)

# The threshold and scale factor parameters come from the documentation: https://data.globalecology.unh.edu/data/GOSIF_v2/Fair_Data_Use_Policy_and_Readme_GOSIF_v2.pdf
# 32767 = water bodies, 32766 = ice/snow
GOSIF_DATA_THRESH = 32765
# This value tells our code the conversion between pixel values in the GeoTIFF images to units of W/m^2/sr/μm
GOSIF_SCALE_FACTOR = 0.0001

# EASE-Grid 2.0 Global (9 km) projection used by SMAP L4 x/y coordinates (meters).
EASE2_GLOBAL_EPSG = "EPSG:6933"

# The SWDI's field capacity and wilting point are derived empirically, per grid
# cell over the 2015-2025 SMAP L4 record in our VDS. This replaces the earlier
# approach of reading them from the SPL4SMLM land-model constants file, which is
# not a valid basis for this index. Field capacity is the high percentile and
# wilting point is the low percentile.
FC_PERCENTILE = 0.95
WP_PERCENTILE = 0.05


def get_read_window(
        geotiff_path: str,
        west: float,
        south: float,
        east: float,
        north: float
) -> Window:
    with rasterio.open(geotiff_path) as src:
        read_window = from_bounds(west, south, east, north, src.transform)
        read_window = read_window.round_offsets().round_lengths()
    return read_window


def read_roi(path: str, read_window: Window) -> np.ndarray:
    """Read the ROI, mask non-data (water/ice/fill), scale to physical units."""
    with rasterio.open(path) as src:
        arr = src.read(1, window=read_window).astype("float64")
    arr[arr > GOSIF_DATA_THRESH] = np.nan
    return arr * GOSIF_SCALE_FACTOR


def write_step_geotiff(
        ras_grid: np.ndarray,
        out_path: str,
        crs,
        transform,
        dtype="float64",
) -> None:
    """Write a single-band raster grid to a georeferenced GeoTIFF."""
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=ras_grid.shape[0],
        width=ras_grid.shape[1],
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=np.nan,
    ) as dst:
        dst.write(ras_grid, 1)


def compute_rci(
        z_jy_grid: np.ndarray,
        z_prev_grid: np.ndarray,
        rci_prev_grid: np.ndarray
) -> np.ndarray:
    """Compute the per grid cell SIF-RCI using the formula from notebook 1."""
    neg_anom = z_jy_grid < -0.75                  # Z(j,y) < -0.75       (case 1)
    pos_anom = z_jy_grid > 0.75                   # Z(j,y) >  0.75       (case 2)
    sign_change = (z_prev_grid * z_jy_grid) < 0   # Z(j-1,y)·Z(j,y) < 0  (case 3)

    neg_term = np.sqrt(np.where(neg_anom, np.abs(z_jy_grid) - 0.75, 0.0))
    pos_term = np.sqrt(np.where(pos_anom, z_jy_grid - 0.75, 0.0))

    rci_jy_grid = rci_prev_grid.copy()
    rci_jy_grid = np.where(neg_anom, rci_prev_grid - neg_term, rci_jy_grid)
    rci_jy_grid = np.where(pos_anom, rci_prev_grid + pos_term, rci_jy_grid)
    rci_jy_grid = np.where(sign_change, 0.0, rci_jy_grid)
    return rci_jy_grid


def compute_sif_time_series(
        gosif_geotiffs: list[str],
        clim_dir: str,
        time_series_fname: str,
        west: float,
        south: float,
        east: float,
        north: float,
        raster_dir: str | None = None,
) -> tuple[str, int]:
    prev_grid: np.ndarray | None = None
    # Initialize the arrays for the rows of our CSV
    # The variable names correspond to what is mentioned in the description above
    dates: list[datetime] = []
    sif_jy: list[float] = []
    mean_sif_j: list[float] = []
    sif_zjy: list[float] = []
    z_jy: list[float] = []
    rci_jy: list[float] = []

    read_window = get_read_window(gosif_geotiffs[0], west, south, east, north)

    # Capture the CRS and windowed transform once so each per-step RCI grid can
    # be written out as a georeferenced GeoTIFF aligned to the read window.
    ras_crs = None
    ras_transform = None
    if raster_dir is not None:
        os.makedirs(raster_dir, exist_ok=True)
        with rasterio.open(gosif_geotiffs[0]) as src:
            ras_crs = src.crs
            ras_transform = src.window_transform(read_window)

    # Recursive state carried between time windows for RCI.
    # RCI(j0, y) = 0 and Z(j0, y) = 0 everywhere
    # For simplicity, our incon j0 = DOY 1
    rci_prev_grid: np.ndarray = np.zeros((read_window.height, read_window.width))
    z_prev_grid: np.ndarray = np.zeros((read_window.height, read_window.width))

    for j, geotiff in enumerate(gosif_geotiffs[1:], start=1):
        # Parse the date from the filename, e.g. GOSIF_2017073.tif = DOY 73
        yr = os.path.splitext(os.path.basename(geotiff))[0][-7:-3]
        doy = os.path.splitext(os.path.basename(geotiff))[0][-3:]
        dates.append(datetime.strptime(f"{yr}{doy}", "%Y%j")) # noqa: DTZ007

        sif_grid = read_roi(geotiff, read_window)
        if j == 1:
            prev_grid = read_roi(gosif_geotiffs[j-1], read_window)
        # Get the SIF increment at j
        dsif_grid = sif_grid - prev_grid
        # Set the current raster to the previous for the next iteration
        prev_grid = sif_grid

        # Get the climatology input file produced by the appendix notebook
        clim_input = f"{clim_dir}/GOSIF_dSIF_clim_{doy}.tif"
        with rasterio.open(clim_input) as clim_src:
            mean_dsif_band = clim_src.read(1).astype(float)
            std_dsif_band = clim_src.read(2).astype(float)
            mean_sif_band = clim_src.read(3).astype(float)
            std_sif_band = clim_src.read(4).astype(float)

        # IMPORTANT: In sif_zjy_grid, we get the Z-Score of SIF. This effectively summarizes
        # the absolute SIF anomaly compared to climatology.
        # zjy_grid is the "SIF Standardized Anomaly" used for calculating SIF-RCI, it is the
        # Z-Score of the CHANGE in SIF between time steps. This summarizes the departure from
        # phenological trend in the vegetation.
        sif_zjy_grid = (sif_grid - mean_sif_band) / std_sif_band
        z_jy_grid = (dsif_grid - mean_dsif_band) / std_dsif_band

        rci_jy_grid = compute_rci(z_jy_grid, z_prev_grid, rci_prev_grid)
        z_prev_grid = z_jy_grid
        rci_prev_grid = rci_jy_grid

        rci_masked = np.where(np.isnan(z_jy_grid), np.nan, rci_jy_grid)

        # Optionally save the non-spatially-averaged RCI grid for this step.
        # TO DO: I may make the write_step_geotiff function more modular to save
        # the other metrics in other bands of the geotiff.
        if raster_dir is not None:
            date = dates[-1]
            out_path = os.path.join(
                raster_dir,
                f"sif_rci_{date.year}_{date.month:02d}_{date.day:02d}.tif",
            )
            write_step_geotiff(rci_masked, out_path, ras_crs, ras_transform)

        # Compute the spatial average at the end
        sif_jy.append(float(np.nanmean(sif_grid)))
        mean_sif_j.append(float(np.nanmean(mean_sif_band)))
        sif_zjy.append(float(np.nanmean(sif_zjy_grid)))
        z_jy.append(float(np.nanmean(z_jy_grid)))
        rci_jy.append(float(np.nanmean(rci_masked)))

    # Save the output as a CSV so it can be used in the next notebook
    csv_path = os.path.join("data", time_series_fname)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "sif", "mean_sif", "sif_zscore", "dsif_zscore", "sif_rci"])
        # We have 4 sig figs from the source data
        writer.writerows(
            zip(
                [d.strftime("%Y-%m-%d") for d in dates],
                [f"{sjy:.4f}" for sjy in sif_jy],
                [f"{msj:.4f}" for msj in mean_sif_j],
                [f"{szj:.4f}" for szj in sif_zjy],
                [f"{zjy:.4f}" for zjy in z_jy],
                [f"{rjy:.4f}" for rjy in rci_jy],
            )
        )

    return csv_path, len(dates)


def detect_flash_drought_sif(
        raster_dir: str,
        output_dir: str,
        time_series_csv: str,
        threshold: float = -0.5,
        n_steps: int = 3,
) -> str:
    """Flag per-cell flash drought from the SIF-RCI raster series.

    Reads the "sif_rci_{year}_{month:02d}_{day:02d}.tif" rasters written by
    :func:`compute_sif_time_series` and applies a rule to each grid cell: a
    flash drought is detected at a time step when that step and the `n_steps`
    - 1 immediately preceding steps all have a SIF-RCI value below `threshold`.
    The chronological filename convention means a lexical sort of the rasters
    is also a temporal sort.

    One GeoTIFF is written per time step to `output_dir` (same filename with a
    "fd_sifrci_" prefix), carrying the source raster's CRS and transform, where
    1 marks a detection and 0 marks no detection. The earliest `n_steps` - 1
    steps lack enough history to satisfy the rule and are therefore all 0.

    The fraction of valid (non-NaN) grid cells flagged at each step is written
    back into the existing `time_series_csv` as a new "fd_percent" column,
    matched to each row by its date so it stays aligned with the other columns.

    Arguments:
        raster_dir (str): Directory holding the SIF-RCI GeoTIFFs.
        output_dir (str): Directory to write the detection GeoTIFFs to.
        time_series_csv (str): Path to the existing time series CSV (with a
            leading "date" column) to add the "fd_percent" column to.
        threshold (float): SIF-RCI value a cell must fall below to count toward
            a detection.
        n_steps (int): Number of consecutive steps (including the current one)
            that must be below `threshold` to flag a detection.

    Returns:
        str: The output directory path.
    """
    paths = sorted(glob.glob(os.path.join(raster_dir, "sif_rci_*.tif")))
    os.makedirs(output_dir, exist_ok=True)

    # Rolling buffer of the last `n_steps` "below threshold" masks so detection
    # only needs each raster in memory once, not the whole stack.
    recent_below: list[np.ndarray] = []
    # Percent of valid cells flagged at each step, keyed by "%Y-%m-%d" date so
    # it can be merged into the CSV by row rather than relying on row order.
    fd_percent_by_date: dict[str, float] = {}

    with open(time_series_csv, newline="") as f:
        reader = csv.DictReader(f)
        time_series_dates = [row["date"] for row in reader]

    for path in paths:
        # Reconstruct the "%Y-%m-%d" date from "sif_rci_{year}_{month}_{day}".
        year, month, day = os.path.splitext(os.path.basename(path))[0].split("_")[-3:]
        if f"{year}-{month}-{day}" not in time_series_dates:
            continue

        with rasterio.open(path) as src:
            rci_grid = src.read(1).astype("float64")
            crs = src.crs
            transform = src.transform

        # NaN (water/ice/fill) compares False, so it never counts as a detection.
        recent_below.append(rci_grid < threshold)
        recent_below = recent_below[-n_steps:]

        if len(recent_below) == n_steps:
            detection = np.logical_and.reduce(recent_below)
        else:
            detection = np.zeros(rci_grid.shape, dtype=bool)

        # Percent over valid land cells only, water/ice/fill (NaN) don't get flagged.
        n_valid = int(np.count_nonzero(~np.isnan(rci_grid)))
        fd_percent = 100.0 * int(np.count_nonzero(detection)) / n_valid if n_valid else 0.0

        fd_percent_by_date[f"{year}-{month}-{day}"] = fd_percent

        out_name = os.path.basename(path).replace("sif_rci_", "fd_sifrci_", 1)
        out_path = os.path.join(output_dir, out_name)
        write_step_geotiff(detection.astype("float32"), out_path, crs, transform, dtype="float32")

    _add_fd_percent_column(time_series_csv, fd_percent_by_date)

    return output_dir


def detect_flash_drought_swdi(
        raster_dir: str,
        output_dir: str,
        time_series_csv: str,
        drop_threshold: float = 2.0,
        abs_threshold: float = -5.0,
        n_lookback: int = 10,
) -> str:
    """Flag per-cell flash drought from the SWDI raster series.

    Reads the "swdi_{year}_{month:02d}_{day:02d}.tif" rasters written by
    :func:`_save_swdi_rasters` and applies two conditions to each grid cell:

    1. The SWDI value has dropped by at least `drop_threshold` points over
       the preceding `n_lookback` rasters (at a 3-day cadence, 10 rasters
       ≈ 30 days).
    2. The current SWDI value is at or below `abs_threshold`.

    Once a cell is positively detected, it remains detected until its SWDI
    rises above `abs_threshold`, at which point the cell reverts to a
    no-detection state and both conditions must be freshly satisfied to
    re-trigger. The chronological filename convention means a lexical sort
    is also a temporal sort.

    One GeoTIFF is written per time step to `output_dir` (same filename with
    a "fd_swdi_" prefix), carrying the source raster's CRS and transform,
    where 1 marks a detection and 0 marks no detection. The earliest
    `n_lookback` steps lack enough history and are all 0.

    The fraction of valid (non-NaN) grid cells flagged at each step is
    written back into the existing `time_series_csv` as a new "fd_percent"
    column, matched to each row by its date.

    Arguments:
        raster_dir (str): Directory holding the SWDI GeoTIFFs.
        output_dir (str): Directory to write the detection GeoTIFFs to.
        time_series_csv (str): Path to the existing time series CSV (with a
            leading "date" column) to add the "fd_percent" column to.
        drop_threshold (float): Minimum decrease in SWDI (in index points)
            over the lookback window required to trigger a new detection.
        abs_threshold (float): SWDI value a cell must be at or below to
            satisfy condition 2, and to sustain an existing detection.
        n_lookback (int): Number of preceding rasters over which the drop is
            measured. With a 3-day cadence, 10 rasters span 30 days.

    Returns:
        str: The output directory path.
    """
    paths = sorted(glob.glob(os.path.join(raster_dir, "swdi_*.tif")))
    os.makedirs(output_dir, exist_ok=True)

    # Rolling buffer of n_lookback + 1 grids so that at step t the oldest
    # entry is exactly n_lookback steps back and the newest is t, giving a
    # drop window of n_lookback steps.
    recent_grids: list[np.ndarray] = []
    # Persistent detection mask carried forward between steps.
    persistent_detection: np.ndarray | None = None
    fd_percent_by_date: dict[str, float] = {}

    with open(time_series_csv, newline="") as f:
        reader = csv.DictReader(f)
        time_series_dates = [row["time"] for row in reader]

    for path in paths:
        year, month, day = os.path.splitext(os.path.basename(path))[0].split("_")[-3:]
        if f"{year}-{month}-{day}" not in time_series_dates:
            continue

        with rasterio.open(path) as src:
            swdi_grid = src.read(1).astype("float64")
            crs = src.crs
            transform = src.transform

        recent_grids.append(swdi_grid)
        recent_grids = recent_grids[-(n_lookback + 1):]

        # NaN comparisons evaluate to False, so water/fill cells never satisfy
        # either condition and can never be flagged.
        below_threshold = swdi_grid <= abs_threshold

        if len(recent_grids) == n_lookback + 1:
            # drop is negative when SWDI has fallen, rule1 is True when the
            # magnitude of the fall meets the threshold.
            drop = recent_grids[-1] - recent_grids[0]
            rule1 = drop <= -drop_threshold
            new_trigger = rule1 & below_threshold

            if persistent_detection is None:
                persistent_detection = new_trigger
            else:
                # Sustain existing detections that remain below the threshold
                persistent_detection = (persistent_detection | new_trigger) & below_threshold
        else:
            # Not enough history for a new trigger
            if persistent_detection is None:
                persistent_detection = np.zeros(swdi_grid.shape, dtype=bool)
            else:
                persistent_detection = persistent_detection & below_threshold

        # NaN guard: fill cells should never appear as detections regardless
        # of how the persistence mask evolves.
        assert persistent_detection is not None
        detection = np.where(np.isnan(swdi_grid), False, persistent_detection)

        n_valid = int(np.count_nonzero(~np.isnan(swdi_grid)))
        fd_percent = 100.0 * int(np.count_nonzero(detection)) / n_valid if n_valid else 0.0

        fd_percent_by_date[f"{year}-{month}-{day}"] = fd_percent

        out_name = os.path.basename(path).replace("swdi_", "fd_swdi_", 1)
        out_path = os.path.join(output_dir, out_name)
        write_step_geotiff(
            detection.astype("float32"), out_path, crs, transform, dtype="float32"
        )

    _add_fd_percent_column(time_series_csv, fd_percent_by_date)

    return output_dir


def _add_fd_percent_column(
        time_series_csv: str,
        fd_percent_by_date: dict[str, float],
) -> None:
    """Add an "fd_percent" column to an existing time series CSV.

    Rows are matched to their detection percent by the leading "date" column so
    the new values stay aligned with the existing rows regardless of order.
    """
    with open(time_series_csv, newline="") as f:
        rows = list(csv.reader(f))

    header, *data_rows = rows
    overwrite = False
    if "fd_percent" not in header:
        header.append("fd_percent")
    else:
        overwrite = True
    for row in data_rows:
        fd_percent = fd_percent_by_date.get(row[0])
        if overwrite:
            row.pop()
        row.append(f"{fd_percent:.2f}" if fd_percent is not None else "")

    with open(time_series_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data_rows)


# Change the number of workers to meet the capabilities of your own computer if needed
def create_dask_cluster(
        n_workers: int = 8,
) -> tuple[Client, LocalCluster, bool]:

    # Reuse an existing local cluster if one is already running so repeated
    # calls don't spin up (and leak) a new LocalCluster each time. Dask
    # registers every Client as the global/default client, so querying for the
    # current one tells us whether a cluster is already up. The returned flag
    # reports whether we created the cluster, so callers know whether it is
    # theirs to shut down.
    try:
        client = Client.current()
    except ValueError:
        print("Creating new local Dask client")
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=1,
            silence_logs=logging.ERROR)

        client = Client(cluster)
        return (client, cluster, True)
    else:
        print("Reusing existing local Dask client")
        return (client, client.cluster, False) # type: ignore


def silence_worker_warnings() -> None:
    warnings.filterwarnings("ignore")
    for name in ["distributed", "xarray", "py.warnings", "fsspec", "h5netcdf", "h5py"]:
        logging.getLogger(name).setLevel(logging.ERROR)


def open_virtual_dataset(
        ref_url: str,
) -> tuple[xr.Dataset, Client, LocalCluster, bool]:
    client, cluster, created = create_dask_cluster()
    client.run(silence_worker_warnings)

    earthaccess.login()
    daac_fs = earthaccess.get_fsspec_https_session()

    fs = fsspec.filesystem(
        "reference",
        fo=ref_url,
        remote_protocol="https",
        asynchronous=True,
        remote_options={"asynchronous": True, **daac_fs.storage_options},
    )

    store = zarr.storage.FsspecStore(fs, read_only=True) # type: ignore
    ds = xr.open_zarr(store, consolidated=False)
    return ds, client, cluster, created


def latlon_bbox_to_ease(
    bbox: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Convert a lat/lon bounding box to EASE-Grid 2.0 Global x/y bounds.

    The SMAP L4 x/y coordinates are in meters in the EASE-Grid 2.0
    Global projection (EPSG:6933), so a geographic bounding box must be
    reprojected before it can be used to index the grid. EPSG:6933 is a
    cylindrical equal-area projection, so x depends only on longitude and
    y only on latitude.

    Arguments:
        bbox: (west, south, east, north) in degrees (lon/lat, EPSG:4326).

    Returns:
        (x_min, y_min, x_max, y_max) in meters (EPSG:6933).
    """
    west, south, east, north = bbox
    transformer = Transformer.from_crs("EPSG:4326", EASE2_GLOBAL_EPSG, always_xy=True)
    xs, ys = transformer.transform([west, east, west, east], [south, south, north, north])
    return min(xs), min(ys), max(xs), max(ys)


def _bounds_slice(coord: xr.DataArray, lo: float, hi: float) -> slice:
    """Build a slice from lo to hi that respects a coordinate's order.

    xarray label slicing follows the coordinate's stored direction, and the
    SMAP L4 y coordinate is descending (north to south), so the slice bounds
    must be reversed for descending coordinates.
    """
    if float(coord[0]) > float(coord[-1]):
        return slice(hi, lo)
    return slice(lo, hi)


def _select_bbox(
    obj: XarrayObj,
    bbox: tuple[float, float, float, float],
) -> XarrayObj:
    """Select the x/y cells of a SMAP L4 grid falling inside a lat/lon box.

    Both the soil moisture and the land-model constants ride on the same
    EASE-Grid 2.0 cells, so selecting each with this shared helper guarantees
    their subsets carry identical x/y coordinates and align cell-for-cell.

    Raises:
        ValueError: If the bounding box does not overlap the dataset grid.
    """
    x_min, y_min, x_max, y_max = latlon_bbox_to_ease(bbox)
    subset = obj.sel(
        x=_bounds_slice(obj.x, x_min, x_max),
        y=_bounds_slice(obj.y, y_min, y_max),
    )
    if subset.sizes["x"] == 0 or subset.sizes["y"] == 0:
        msg = f"Bounding box {bbox} does not overlap the dataset grid."
        raise ValueError(msg)
    return subset


def _read_record_in_blocks(
    sm: xr.DataArray,
    retries: int = 2,
    retry_wait: float = 2.0,
    context: str = "the FC/WP percentiles",
) -> xr.DataArray:
    """Materialise `sm` over its whole time span, tolerating unreachable references.

    If a chunk (aka a month) is repeatedly unreadable it gets skipped which will add uncertainty to
    the result but it's a tradeoff to get the large amount of data we're reading from the VDS to
    process without throwing an exception.

    Arguments:
        sm (xr.DataArray): Lazy root-zone soil moisture with (time, y, x) dims.
        retries (int): Extra attempts per month after the first (so ``retries=2``
            means up to three tries) before the month is skipped.
        retry_wait (float): Base seconds to wait between attempts. The wait grows
            with each attempt to let a transient outage clear.
        context (str): Short phrase naming what the read feeds, used in the warning
            messages (e.g. "the FC/WP percentiles" or "the SWDI time series").

    Returns:
        xr.DataArray: The successfully read months concatenated and sorted in
        time, held in memory.

    Raises:
        RuntimeError: If no month could be read at all.
    """
    # Group by calendar month (YYYYMM) so a single bad reference costs only that
    # month, and so no single read schedules all ~30k chunks at once.
    month_id = sm["time"].dt.year * 100 + sm["time"].dt.month
    blocks: list[xr.DataArray] = []
    skipped = 0
    for _, block in sm.groupby(month_id):
        label = str(block["time"].values[0])[:7]
        for attempt in range(retries + 1):
            try:
                blocks.append(block.compute())
                break
            except Exception as exc:  # noqa: BLE001 - tolerate any read failure
                if attempt < retries:
                    time.sleep(retry_wait * (attempt + 1))
                    continue
                skipped += 1
                warnings.warn(
                    f"Skipping {label} while reading soil moisture for {context} "
                    f"(unreadable after {retries + 1} tries): {exc!r}",
                    stacklevel=2,
                )
    if not blocks:
        msg = f"Could not read any soil moisture for {context}."
        raise RuntimeError(msg)
    if skipped:
        warnings.warn(
            f"{context} computed with {skipped} month(s) skipped due to "
            "unreachable references.",
            stacklevel=2,
        )
    return xr.concat(blocks, dim="time").sortby("time")


def _field_capacity_wilting_point(
    sm: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Derive per-cell field capacity and wilting point from soil-moisture percentiles.

    Field capacity (the long-term "wet" level) and wilting point (the "dry"
    level) are estimated at each grid cell as the ``FC_PERCENTILE`` and
    ``WP_PERCENTILE`` quantiles of the root-zone soil moisture taken over the
    entire time span of ``sm``, 2015-2025. The idea is that this record is long
    enough for stable estimates of these distribution tails.
    Cells that are entirely fill/water reduce to NaN.

    The record is pulled into memory a month at a time by
    :func:`_read_record_in_blocks` so a single unreachable granule does not throw an
    exception for the whole 90+ minute process. The quantiles are then taken over the
    time series in memory. This function is meant to work with small spatial regions
    instead of global extent.

    Arguments:
        sm (xr.DataArray): Root-zone soil moisture with (time, y, x) dimensions,
            already subset to the region of interest.

    Returns:
        tuple[xr.DataArray, xr.DataArray]: The (field_capacity, wilting_point)
        DataArrays on the (y, x) grid of ``sm``, in the same units as ``sm``.
    """
    sm = _read_record_in_blocks(sm)
    quantiles = sm.quantile([WP_PERCENTILE, FC_PERCENTILE], dim="time", skipna=True)
    field_capacity = (
        quantiles.sel(quantile=FC_PERCENTILE)
        .drop_vars("quantile")
        .rename("field_capacity")
    )
    wilting_point = (
        quantiles.sel(quantile=WP_PERCENTILE)
        .drop_vars("quantile")
        .rename("wilting_point")
    )
    return field_capacity, wilting_point


def _write_fc_wp_geotiff(
    field_capacity: xr.DataArray,
    wilting_point: xr.DataArray,
    out_path: str,
    bbox: tuple[float, float, float, float],
    variable: str,
) -> None:
    """Cache per-cell field capacity and wilting point to a two-band GeoTIFF.

    Band 1 is field capacity and band 2 is wilting point in m3/m3.  The
    box, percentile levels and source variable are recorded as tags.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # Orient north-up (row 0 = northernmost) and west-to-east so the array rows
    # and columns line up with the affine transform -- and so a reader can
    # reconstruct the coordinates simply by sorting the grid the same way.
    field_capacity = field_capacity.sortby("y", ascending=False).sortby("x").transpose("y", "x")
    wilting_point = wilting_point.sortby("y", ascending=False).sortby("x").transpose("y", "x")
    transform = ease2_grid_transform(field_capacity.x.values, field_capacity.y.values)

    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=field_capacity.sizes["y"],
        width=field_capacity.sizes["x"],
        count=2,
        dtype="float64",
        crs=EASE2_GLOBAL_EPSG,
        transform=transform,
        nodata=np.nan,
    ) as dst:
        dst.write(field_capacity.values, 1)
        dst.write(wilting_point.values, 2)
        dst.set_band_description(1, "field_capacity")
        dst.set_band_description(2, "wilting_point")
        dst.update_tags(
            bbox=",".join(repr(float(b)) for b in bbox),
            fc_percentile=repr(FC_PERCENTILE),
            wp_percentile=repr(WP_PERCENTILE),
            variable=variable,
        )


def _read_fc_wp_geotiff(
    cache_path: str,
    sm: xr.DataArray,
    bbox: tuple[float, float, float, float],
    variable: str,
) -> tuple[xr.DataArray, xr.DataArray] | None:
    """Read cached field capacity and wilting point, or None if the cache does not fit.

    The cache is reused only when its stored box, percentile levels, source
    variable and grid shape all match the current request, otherwise None is
    returned so the caller recomputes (and overwrites) it. The two bands are
    wrapped as DataArrays carrying ``sm``'s own y/x coordinate values (sorted
    into the north-up, west-to-east layout the file was written in) so they align
    cell-for-cell with the soil moisture when the SWDI is formed.
    """
    with rasterio.open(cache_path) as src:
        if src.count < 2:
            return None

        tags = src.tags()
        expected = {
            "fc_percentile": repr(FC_PERCENTILE),
            "wp_percentile": repr(WP_PERCENTILE),
            "variable": variable,
        }
        if any(tags.get(key) != value for key, value in expected.items()):
            return None
        try:
            cached_bbox = [float(v) for v in tags.get("bbox", "").split(",")]
        except ValueError:
            return None
        if len(cached_bbox) != 4 or not np.allclose(cached_bbox, bbox):
            return None

        y = np.sort(sm["y"].values)[::-1]
        x = np.sort(sm["x"].values)
        if (src.height, src.width) != (y.size, x.size):
            return None

        field_capacity = src.read(1).astype("float64")
        wilting_point = src.read(2).astype("float64")

    coords = {"y": ("y", y), "x": ("x", x)}
    return (
        xr.DataArray(field_capacity, dims=("y", "x"), coords=coords, name="field_capacity"),
        xr.DataArray(wilting_point, dims=("y", "x"), coords=coords, name="wilting_point"),
    )


def _load_or_compute_fc_wp(
    sm: xr.DataArray,
    cache_path: str | None,
    bbox: tuple[float, float, float, float],
    variable: str,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Return per-cell field capacity and wilting point, using the cache when usable.

    When ``cache_path`` names an existing two-band GeoTIFF that matches this box,
    the constants are read straight from it. Otherwise they are derived from the
    soil-moisture percentiles (:func:`_field_capacity_wilting_point`),
    materialised, and -- when ``cache_path`` is given -- written out so later runs
    can skip the expensive percentile computation.
    """
    if cache_path is not None and os.path.exists(cache_path):
        cached = _read_fc_wp_geotiff(cache_path, sm, bbox, variable)
        if cached is not None:
            return cached

    field_capacity, wilting_point = _field_capacity_wilting_point(sm)
    field_capacity, wilting_point = field_capacity.compute(), wilting_point.compute()

    if cache_path is not None:
        _write_fc_wp_geotiff(field_capacity, wilting_point, cache_path, bbox, variable)

    return field_capacity, wilting_point


def ease2_grid_transform(x: np.ndarray, y: np.ndarray):
    """Affine transform for a north-up EASE-Grid 2.0 raster from cell centers.

    The SMAP L4 x/y coordinates are the cell centers (meters) of a regular
    grid, so the pixel size is the coordinate spacing and the raster origin is
    the outer corner of the north-west cell (half a pixel beyond the extreme
    centers).  min/max are used so the transform is correct regardless of
    whether x/y are stored ascending or descending. Callers must orient the
    array itself north-up (row 0 = northernmost row) to match.
    """
    xres = abs(float(x[1] - x[0]))
    yres = abs(float(y[1] - y[0]))
    west = float(x.min()) - xres / 2.0
    north = float(y.max()) + yres / 2.0
    return from_origin(west, north, xres, yres)


def swdi_timeseries(
    ds: xr.Dataset,
    bbox: tuple[float, float, float, float],
    freq: str = "3D",
    variable: str = "sm_rootzone",
    start: str | None = None,
    stop: str | None = None,
    raster_dir: str | None = None,
    fc_wp_path: str | None = None,
) -> xr.DataArray:
    """Compute a box-averaged Soil Water Deficit Index (SWDI) time series.
    The SWDI is computed per cell and then spatially averaged over the bbox.

    Field capacity and wilting point are derived per cell from percentiles of
    the root-zone soil moisture over the *whole* time span of ``ds``. That
    percentile calculation is expensive, so the result is cached to a two-band
    GeoTIFF at ``fc_wp_path`` (band 1 field capacity, band 2 wilting point) and
    reused on later runs (see :func:`_load_or_compute_fc_wp`).

    Field capacity and wilting point are constant in time, so aggregating the
    soil moisture to `freq` windows before forming the (linear) SWDI is
    equivalent to forming it first and then aggregating. The soil moisture is
    resampled first so each window's SWDI is built from that window's moisture.

    Arguments:
        ds (xr.Dataset): The SMAP L4 virtual dataset. Its full time span is used
            to derive field capacity and wilting point.
        bbox: (west, south, east, north) in degrees (lon/lat).
        freq (str): Pandas offset alias for the temporal aggregation window
            ("3D" = 3-day means).
        variable (str): The root-zone soil moisture variable to use.
        start (str | None): Optional start date (e.g. "2019" or "2019-01-01")
            for the returned series. If None, begins at the start of the dataset.
        stop (str | None): Optional end date (e.g. "2019" or "2019-12-31"),
            inclusive. If None, runs to the end of the dataset.
        raster_dir (str | None): Optional directory in which to save the
            non-spatially-averaged per-cell SWDI grid for each time step as a
            georeferenced GeoTIFF (EPSG:6933), named
            "swdi_{year}_{month}_{day}.tif" (ordered so the files sort
            chronologically). If None, no rasters are written.
        fc_wp_path (str | None): Optional path to the two-band field capacity /
            wilting point GeoTIFF cache. Read from when it already matches this
            box, otherwise (re)computed and written. If None, the constants are
            computed without being cached.

    Returns:
        xr.DataArray: A 1-D DataArray of the box-averaged SWDI indexed by time.

    Raises:
        ValueError: If the bounding box does not overlap the dataset grid.
    """
    subset = _select_bbox(ds, bbox)
    sm_full = subset[variable]

    # Per-cell field capacity and wilting point from the full-record percentiles
    # (cached to GeoTIFF).  Their x/y coordinates come from the same box subset,
    # so xarray broadcasts them against the (time, y, x) soil moisture
    # cell-for-cell.
    field_capacity, wilting_point = _load_or_compute_fc_wp(
        sm_full, fc_wp_path, bbox, variable
    )

    sm = sm_full
    if start is not None or stop is not None:
        sm = sm.sel(time=slice(start, stop))
    sm = _read_record_in_blocks(sm, context="the SWDI time series")
    sm = sm.resample(time=freq).mean()

    swdi = (sm - field_capacity) / (field_capacity - wilting_point) * 10.0

    if raster_dir is not None:
        # Writing the rasters already materialises the full (time, y, x) grid
        # in memory, so reuse it for the spatial average instead of forcing a
        # second (slow, network-bound) read of the virtualised dataset.
        swdi = _save_swdi_rasters(swdi, raster_dir)

    # Average the SWDI (not the soil moisture) across the box.
    return swdi.mean(dim=("x", "y")).rename("swdi")


def _save_swdi_rasters(swdi: xr.DataArray, raster_dir: str) -> xr.DataArray:
    """Write each SWDI time step to a georeferenced GeoTIFF in `raster_dir`.

    The grid is oriented north-up (y descending) so its rows match the affine
    transform, then materialised once. The computed in-memory grid is returned
    so the caller can spatially average it without re-reading the (virtualised,
    network-bound) source data.
    """
    os.makedirs(raster_dir, exist_ok=True)

    # Orient north-up (row 0 = northernmost) and materialise the lazy grid so
    # every time step is computed a single time.
    swdi = swdi.sortby("y", ascending=False).compute()
    transform = ease2_grid_transform(swdi.x.values, swdi.y.values)

    for step in swdi.transpose("time", "y", "x"):
        date = step.time.values.astype("datetime64[s]").item()
        out_path = os.path.join(
            raster_dir,
            f"swdi_{date.year}_{date.month:02d}_{date.day:02d}.tif",
        )
        write_step_geotiff(step.values, out_path, EASE2_GLOBAL_EPSG, transform)

    return swdi


def compute_swdi_timeseries(
        start_date: str,
        stop_date: str,
        time_series_fname: str,
        bbox: tuple[float, float, float, float],
        ref_url: str = "https://its-live-data.s3-us-west-2.amazonaws.com/test-space/vds/SPL4SMGP.parquet",
        raster_dir: str | None = None,
        fc_wp_path: str | None = None,
) -> str:
    ds, client, cluster, created = open_virtual_dataset(ref_url)
    try:
        if fc_wp_path is None:
            stem = os.path.splitext(os.path.basename(time_series_fname))[0]
            fc_wp_path = os.path.join("inputs", f"{stem}_fc_wp.tif")

        swdi_ts = swdi_timeseries(
            ds, bbox,
            start=start_date, stop=stop_date,
            raster_dir=raster_dir, fc_wp_path=fc_wp_path,
        )
        csv_path = os.path.join("data", time_series_fname)

        swdi_ts.to_dataframe().to_csv(csv_path)
        return csv_path
    finally:
        # Only tear down the cluster if this workflow created it, so a client
        # the user already had running isn't shut down out from under them.
        if created:
            client.close()
            cluster.close()