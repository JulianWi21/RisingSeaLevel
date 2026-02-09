"""
Download elevation data (SRTM) and a country boundary for sea level rise visualization.
Uses NASA SRTM data via direct tile downloads and Natural Earth boundaries.
"""

import os
import re
import argparse
import urllib.request
import zipfile
import io
import numpy as np
import rasterio
import pandas as pd
from rasterio.merge import merge
from rasterio.mask import mask as rasterio_mask
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
import geopandas as gpd
from shapely.geometry import box as shapely_box
from shapely.prepared import prep as shapely_prep
from affine import Affine
try:
    from scipy.ndimage import binary_dilation as scipy_binary_dilation
except Exception:
    scipy_binary_dilation = None

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)
DEFAULT_DEM_SIZE = "7680x4320"
DEFAULT_WATER = True
SPECIAL_REGIONS = {
    # Approximate Europe mainland bbox to avoid overseas territories
    "europe": {"continent": "Europe", "bbox": (-25.0, 34.0, 45.0, 75.0), "include": ["Turkey", "Türkiye"]},
    "europa": {"continent": "Europe", "bbox": (-25.0, 34.0, 45.0, 75.0), "include": ["Turkey", "Türkiye"]},
}

def slugify(name):
    value = name.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "country"


def select_country(world, country_name):
    name_lower = country_name.strip().lower()
    candidate_cols = ["NAME", "NAME_EN", "ADMIN", "SOVEREIGNT"]
    for col in candidate_cols:
        if col in world.columns:
            series = world[col].fillna("").astype(str).str.lower()
            matches = world[series == name_lower]
            if not matches.empty:
                return matches
    for col in candidate_cols:
        if col in world.columns:
            series = world[col].fillna("").astype(str).str.lower()
            matches = world[series.str.contains(name_lower)]
            if not matches.empty:
                print(f"Warning: using partial match from column {col}")
                return matches
    raise ValueError(f"Country '{country_name}' not found in Natural Earth dataset.")


def select_region(world, region_key):
    spec = SPECIAL_REGIONS[region_key]
    world_4326 = world.to_crs("EPSG:4326")
    gdf = world_4326
    if "continent" in spec:
        col = "CONTINENT" if "CONTINENT" in gdf.columns else None
        if not col:
            raise ValueError("Natural Earth dataset missing CONTINENT column.")
        target = spec["continent"].lower()
        gdf = gdf[gdf[col].fillna("").astype(str).str.lower() == target].copy()
    if gdf.empty:
        raise ValueError(f"Region '{region_key}' has no matching features.")
    if spec.get("include"):
        extra = []
        candidate_cols = ["NAME", "NAME_EN", "ADMIN", "SOVEREIGNT"]
        for name in spec["include"]:
            name_lower = name.strip().lower()
            matches = None
            for col in candidate_cols:
                if col in world_4326.columns:
                    series = world_4326[col].fillna("").astype(str).str.lower()
                    exact = world_4326[series == name_lower]
                    if not exact.empty:
                        matches = exact
                        break
            if matches is None:
                for col in candidate_cols:
                    if col in world_4326.columns:
                        series = world_4326[col].fillna("").astype(str).str.lower()
                        partial = world_4326[series.str.contains(name_lower)]
                        if not partial.empty:
                            matches = partial
                            break
            if matches is not None and not matches.empty:
                extra.append(matches)
            else:
                print(f"Warning: could not include '{name}' (not found).")
        if extra:
            gdf = gpd.GeoDataFrame(
                pd.concat([gdf] + extra, ignore_index=True), crs=gdf.crs
            )
    if spec.get("bbox"):
        bbox_poly = shapely_box(*spec["bbox"])
        gdf = gpd.clip(gdf, bbox_poly)
        gdf = gdf[~gdf.is_empty]
    if gdf.empty:
        raise ValueError(f"Region '{region_key}' became empty after clipping.")
    return gpd.GeoDataFrame({"geometry": [gdf.geometry.unary_union]}, crs=gdf.crs)


def select_mainland(country_gdf):
    """Pick the largest polygon (by area) as mainland."""
    exploded = country_gdf.explode(index_parts=False).reset_index(drop=True)
    if exploded.empty:
        return country_gdf
    try:
        areas = exploded.to_crs("EPSG:6933").area
    except Exception:
        areas = exploded.area
    idx = areas.idxmax()
    return exploded.loc[[idx]].copy()


def download_country_boundary(country_name, country_slug, country_dir, mainland=False):
    """Download country boundary from Natural Earth 10m countries."""
    boundary_file = os.path.join(country_dir, f"{country_slug}_boundary.gpkg")
    if os.path.exists(boundary_file):
        print(f"{country_name} boundary already exists, loading...")
        return gpd.read_file(boundary_file)

    print("Downloading Natural Earth 10m countries...")
    url = "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip"
    zip_path = os.path.join(country_dir, "ne_10m_admin_0_countries.zip")

    urllib.request.urlretrieve(url, zip_path)
    print(f"Download complete. Extracting {country_name}...")

    world = gpd.read_file(f"zip://{zip_path}")
    region_key = country_slug
    if region_key in SPECIAL_REGIONS:
        if mainland:
            print("Warning: --mainland ignored for regions.")
        country = select_region(world, region_key).copy()
    else:
        country = select_country(world, country_name).copy()
        if mainland:
            country = select_mainland(country)
        elif len(country) > 1:
            country = gpd.GeoDataFrame(
                {"geometry": [country.geometry.unary_union]}, crs=country.crs
            )
    country = country.to_crs("EPSG:4326")
    country.to_file(boundary_file, driver="GPKG")

    # Cleanup
    os.remove(zip_path)
    print(f"{country_name} boundary saved to {boundary_file}")
    return country


def download_srtm_tiles():
    """Download SRTM 3-arc-second tiles covering Germany."""
    # Germany bounds: approximately N47-N55, E005-E015
    tiles_dir = os.path.join(DATA_DIR, "srtm_tiles")
    os.makedirs(tiles_dir, exist_ok=True)

    # SRTM tile naming: each tile covers 1x1 degree, named by SW corner
    # We need tiles from N47E005 to N55E015
    tile_files = []

    for lat in range(47, 56):  # N47 to N55 inclusive
        for lon in range(5, 16):  # E005 to E015 inclusive
            tile_name = f"N{lat:02d}E{lon:03d}"
            hgt_file = os.path.join(tiles_dir, f"{tile_name}.hgt")

            if os.path.exists(hgt_file):
                tile_files.append(hgt_file)
                continue

            # Try CGIAR SRTM v4.1 (90m, no auth needed)
            # These come as 5x5 degree tiles
            # Alternative: use viewfinderpanoramas.org which has free SRTM data
            url = f"https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/TIFF/"

            # We'll use a different source - viewfinderpanoramas
            # which provides HGT files directly
            print(f"  Need tile {tile_name}")

    return tile_files


def download_srtm_via_elevation_api(country_gdf, country_slug, country_dir):
    """
    Download SRTM data for a country using a direct approach.
    Downloads from CGIAR SRTM 90m (5x5 degree tiles).
    """
    merged_file = os.path.join(country_dir, f"{country_slug}_srtm.tif")
    if os.path.exists(merged_file):
        print("SRTM data already exists.")
        return merged_file

    # CGIAR SRTM v4.1 tiles covering Germany
    # Tile numbering: col 38-39, row 01-03 cover Europe including Germany
    # Format: srtm_COL_ROW.zip containing srtm_COL_ROW.tif
    tiles_dir = os.path.join(country_dir, "srtm_tiles")
    os.makedirs(tiles_dir, exist_ok=True)

    # CGIAR tile grid: each tile is 5x5 degrees
    # col = floor((lon + 180) / 5) + 1
    # row = floor((60 - lat) / 5) + 1
    country_4326 = country_gdf.to_crs("EPSG:4326")
    try:
        country_geom = country_4326.geometry.union_all()
    except AttributeError:
        country_geom = country_4326.geometry.unary_union
    minx, miny, maxx, maxy = country_geom.bounds
    orig_miny, orig_maxy = miny, maxy
    # Clamp to SRTM coverage (~60N to ~56S)
    miny = max(miny, -56.0)
    maxy = min(maxy, 60.0)
    if orig_maxy > 60 or orig_miny < -56:
        print("Warning: SRTM coverage is limited (~60N to ~56S). Results may be incomplete.")
    col_min = int((minx + 180) // 5) + 1
    col_max = int((maxx + 180) // 5) + 1
    row_min = int((60 - maxy) // 5) + 1
    row_max = int((60 - miny) // 5) + 1

    if miny > maxy:
        raise RuntimeError("Country is outside SRTM coverage range (~60N to ~56S).")

    prepared = shapely_prep(country_geom)
    tile_coords = []
    for col in range(col_min, col_max + 1):
        lon_min = (col - 1) * 5 - 180
        lon_max = lon_min + 5
        for row in range(row_min, row_max + 1):
            lat_max = 60 - (row - 1) * 5
            lat_min = lat_max - 5
            tile_poly = shapely_box(lon_min, lat_min, lon_max, lat_max)
            if prepared.intersects(tile_poly):
                tile_coords.append((col, row))

    print(f"Tiles to download (intersecting country): {len(tile_coords)}")

    tif_files = []
    for col, row in tile_coords:
        tile_name = f"srtm_{col:02d}_{row:02d}"
        tif_file = os.path.join(tiles_dir, f"{tile_name}.tif")

        if os.path.exists(tif_file):
            print(f"  Tile {tile_name} already exists.")
            tif_files.append(tif_file)
            continue

        # Try multiple CGIAR mirror URLs
        urls = [
            f"https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/TIFF/{tile_name}.zip",
            f"https://data.cgiarcsi.info/srtm/tiles/GeoTIFF/{tile_name}.zip",
        ]

        downloaded = False
        for url in urls:
            try:
                print(f"  Downloading {tile_name} from {url}...")
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                response = urllib.request.urlopen(req, timeout=60)
                zip_data = response.read()

                with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
                    for name in zf.namelist():
                        if name.endswith('.tif'):
                            zf.extract(name, tiles_dir)
                            extracted = os.path.join(tiles_dir, name)
                            if extracted != tif_file:
                                os.rename(extracted, tif_file)
                            break

                tif_files.append(tif_file)
                downloaded = True
                print(f"  {tile_name} downloaded successfully.")
                break
            except Exception as e:
                print(f"  Failed from {url}: {e}")

        if not downloaded:
            print(f"  WARNING: Could not download {tile_name}")

    if not tif_files:
        raise RuntimeError("No SRTM tiles downloaded!")

    # Merge all tiles
    print("Merging SRTM tiles...")
    datasets = [rasterio.open(f) for f in tif_files]
    merged_data, merged_transform = merge(datasets)

    profile = datasets[0].profile.copy()
    profile.update({
        'height': merged_data.shape[1],
        'width': merged_data.shape[2],
        'transform': merged_transform,
        'driver': 'GTiff',
        'compress': 'lzw'
    })

    with rasterio.open(merged_file, 'w', **profile) as dst:
        dst.write(merged_data)

    for ds in datasets:
        ds.close()

    print(f"Merged SRTM saved to {merged_file}")
    return merged_file


def clip_dem_to_country(dem_path, country_gdf, country_slug, country_dir):
    """Clip DEM to country boundary."""
    clipped_file = os.path.join(country_dir, f"{country_slug}_dem_clipped.tif")
    if os.path.exists(clipped_file):
        print("Clipped DEM already exists.")
        return clipped_file

    print("Clipping DEM to country boundary...")
    country_4326 = country_gdf.to_crs("EPSG:4326")

    with rasterio.open(dem_path) as src:
        out_image, out_transform = rasterio_mask(
            src,
            country_4326.geometry,
            crop=True,
            nodata=-9999,
            filled=True
        )
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "nodata": -9999,
            "compress": "lzw"
        })

    with rasterio.open(clipped_file, 'w', **out_meta) as dst:
        dst.write(out_image)

    print(f"Clipped DEM saved to {clipped_file}")
    return clipped_file


def parse_size(value):
    m = re.match(r"^\s*(\d+)\s*x\s*(\d+)\s*$", value.lower())
    if not m:
        raise ValueError("Size must be in the format WIDTHxHEIGHT (e.g. 1920x1080).")
    return int(m.group(1)), int(m.group(2))


def resample_dem(clipped_path, country_slug, country_dir, target_width, target_height, keep_aspect=True):
    """Create a resized DEM clipped to the country."""
    with rasterio.open(clipped_path) as src:
        src_width, src_height = src.width, src.height
        if keep_aspect:
            scale = min(target_width / src_width, target_height / src_height, 1.0)
            new_width = max(1, int(round(src_width * scale)))
            new_height = max(1, int(round(src_height * scale)))
        else:
            new_width, new_height = target_width, target_height

    resized_file = os.path.join(country_dir, f"{country_slug}_{new_width}x{new_height}_dem_clipped.tif")
    if os.path.exists(resized_file):
        print("Resized DEM already exists.")
        return resized_file, new_width, new_height

    print(f"Resampling DEM to {new_width}x{new_height} (target {target_width}x{target_height})...")
    with rasterio.open(clipped_path) as src:
        downsampling = new_width < src.width or new_height < src.height
        data = src.read(
            1,
            out_shape=(new_height, new_width),
            resampling=Resampling.average if downsampling else Resampling.bilinear,
            masked=True,
        )
        scale_x = src.width / new_width
        scale_y = src.height / new_height
        transform = src.transform * Affine.scale(scale_x, scale_y)

        profile = src.profile.copy()
        profile.update({
            "height": new_height,
            "width": new_width,
            "transform": transform,
            "compress": "lzw",
        })

        filled = data.filled(src.nodata if src.nodata is not None else -9999)
        if profile.get("nodata", None) is None:
            profile["nodata"] = src.nodata if src.nodata is not None else -9999

        with rasterio.open(resized_file, "w", **profile) as dst:
            dst.write(filled, 1)

    print(f"Resized DEM saved to {resized_file}")
    return resized_file, new_width, new_height


def download_if_missing(filename, url):
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        print(f"  Downloading {filename}...")
        urllib.request.urlretrieve(url, filepath)
    return filepath


def build_water_raster(dem_path, water_tag, output_dir):
    """Precompute water body raster for a DEM."""
    water_raster_path = os.path.join(output_dir, f"water_bodies_{water_tag}.npy")
    if os.path.exists(water_raster_path):
        print(f"Water body raster already exists: {water_raster_path}")
        return water_raster_path

    with rasterio.open(dem_path) as src:
        dem_transform = src.transform
        dem_shape = (src.height, src.width)
        dem_bounds = src.bounds

    clip_box = shapely_box(dem_bounds.left, dem_bounds.bottom, dem_bounds.right, dem_bounds.top)

    datasets = {
        "rivers_main": ("ne_10m_rivers_lake_centerlines.zip",
                        "https://naciscdn.org/naturalearth/10m/physical/ne_10m_rivers_lake_centerlines.zip", "line"),
        "rivers_europe": ("ne_10m_rivers_europe.zip",
                          "https://naciscdn.org/naturalearth/10m/physical/ne_10m_rivers_europe.zip", "line"),
        "lakes": ("ne_10m_lakes.zip",
                  "https://naciscdn.org/naturalearth/10m/physical/ne_10m_lakes.zip", "polygon"),
        "lakes_europe": ("ne_10m_lakes_europe.zip",
                         "https://naciscdn.org/naturalearth/10m/physical/ne_10m_lakes_europe.zip", "polygon"),
    }

    # Scale dilation iterations by DEM resolution (reference: SRTM ~90m/pixel)
    pixel_deg = abs(dem_transform.a)
    pixel_m = pixel_deg * 111000
    ref_pixel_m = 90.0
    dilation_scale = ref_pixel_m / pixel_m
    print(f"Building water raster for {water_tag} (pixel ~{pixel_m:.0f}m, dilation scale {dilation_scale:.2f})...")
    water_raster = np.zeros(dem_shape, dtype=np.uint8)

    for name, (filename, url, geom_type) in datasets.items():
        filepath = download_if_missing(filename, url)
        try:
            gdf = gpd.read_file(f"zip://{filepath}")
            gdf = gdf.to_crs("EPSG:4326")
            clipped = gpd.clip(gdf, clip_box)
            clipped = clipped[~clipped.is_empty]
            if len(clipped) == 0:
                continue
            print(f"  {name}: {len(clipped)} features")

            if geom_type == "line" and "scalerank" in clipped.columns:
                for width, max_rank in [(3, 3), (2, 6), (1, 12)]:
                    subset = clipped[clipped["scalerank"] <= max_rank]
                    if len(subset) == 0:
                        continue
                    shapes = [(g, 1) for g in subset.geometry if g is not None]
                    layer = rasterize(
                        shapes, out_shape=dem_shape, transform=dem_transform,
                        fill=0, dtype=np.uint8, all_touched=True
                    )
                    scaled_iters = max(0, round((width - 1) * dilation_scale))
                    if scaled_iters > 0 and scipy_binary_dilation is not None:
                        layer = scipy_binary_dilation(layer, iterations=scaled_iters).astype(np.uint8)
                    water_raster = np.maximum(water_raster, layer)
            else:
                shapes = [(g, 1) for g in clipped.geometry if g is not None]
                layer = rasterize(
                    shapes, out_shape=dem_shape, transform=dem_transform,
                    fill=0, dtype=np.uint8, all_touched=True
                )
                water_raster = np.maximum(water_raster, layer)
        except Exception as e:
            print(f"  {name}: ERROR - {e}")

    np.save(water_raster_path, water_raster)
    print(f"Water body raster saved: {water_raster_path}")
    return water_raster_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download SRTM and clip to a country boundary.")
    parser.add_argument("country", nargs="?",
                        help="Country name from Natural Earth (e.g. France).")
    parser.add_argument("--country", dest="country_opt", default=None,
                        help="Country name from Natural Earth (overrides positional).")
    parser.add_argument("--mainland", action="store_true",
                        help="Use largest polygon only (e.g. metropolitan France).")
    parser.add_argument("--dem-size", default=DEFAULT_DEM_SIZE,
                        help="Optional: also write a resized DEM (max size) like 3840x2160 (saved in the country folder).")
    parser.add_argument("--no-dem-size", action="store_true",
                        help="Do not write a resized DEM.")
    parser.add_argument("--dem-force", action="store_true",
                        help="Force exact dem-size (may stretch).")
    parser.add_argument("--water", dest="water", action="store_true", default=DEFAULT_WATER,
                        help="Precompute water body rasters for the clipped DEM(s).")
    parser.add_argument("--no-water", dest="water", action="store_false",
                        help="Do not precompute water body rasters.")
    args = parser.parse_args()

    if args.country_opt and args.country and args.country_opt != args.country:
        print("Warning: both positional and --country provided; using --country.")
    country_name = args.country_opt or args.country or "Germany"
    country_slug = slugify(country_name)
    country_dir = os.path.join(DATA_DIR, country_slug)
    os.makedirs(country_dir, exist_ok=True)

    print("=== Downloading Data for Sea Level Rise Visualization ===\n")
    print(f"Country: {country_name}\n")

    # Step 1: Country boundary
    country = download_country_boundary(country_name, country_slug, country_dir, mainland=args.mainland)
    print(f"{country_name} bounds: {country.total_bounds}\n")

    # Step 2: SRTM elevation data
    dem_path = download_srtm_via_elevation_api(country, country_slug, country_dir)

    # Step 3: Clip to country
    clipped_path = clip_dem_to_country(dem_path, country, country_slug, country_dir)
    resized_path = None
    resized_tag = None
    if args.no_dem_size:
        args.dem_size = None
    if args.dem_size:
        try:
            w, h = parse_size(args.dem_size)
            resized_path, resized_w, resized_h = resample_dem(
                clipped_path, country_slug, country_dir, w, h, keep_aspect=not args.dem_force
            )
            resized_tag = f"{country_slug}_{resized_w}x{resized_h}"
        except ValueError as e:
            print(f"Invalid --dem-size: {e}")
            raise

    if args.water:
        build_water_raster(clipped_path, country_slug, country_dir)
        if resized_path and resized_tag:
            build_water_raster(resized_path, resized_tag, country_dir)

    print("\n=== Data download complete! ===")
    print(f"Clipped DEM: {clipped_path}")
