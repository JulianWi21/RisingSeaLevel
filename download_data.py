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
from rasterio.merge import merge
from rasterio.mask import mask as rasterio_mask
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
import geopandas as gpd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

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


def download_country_boundary(country_name, country_slug):
    """Download country boundary from Natural Earth 10m countries."""
    boundary_file = os.path.join(DATA_DIR, f"{country_slug}_boundary.gpkg")
    if os.path.exists(boundary_file):
        print(f"{country_name} boundary already exists, loading...")
        return gpd.read_file(boundary_file)

    print("Downloading Natural Earth 10m countries...")
    url = "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip"
    zip_path = os.path.join(DATA_DIR, "ne_10m_admin_0_countries.zip")

    urllib.request.urlretrieve(url, zip_path)
    print(f"Download complete. Extracting {country_name}...")

    world = gpd.read_file(f"zip://{zip_path}")
    country = select_country(world, country_name).copy()
    if len(country) > 1:
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


def download_srtm_via_elevation_api(country_bounds, country_slug):
    """
    Download SRTM data for a country using a direct approach.
    Downloads from CGIAR SRTM 90m (5x5 degree tiles).
    """
    merged_file = os.path.join(DATA_DIR, f"{country_slug}_srtm.tif")
    if os.path.exists(merged_file):
        print("SRTM data already exists.")
        return merged_file

    # CGIAR SRTM v4.1 tiles covering Germany
    # Tile numbering: col 38-39, row 01-03 cover Europe including Germany
    # Format: srtm_COL_ROW.zip containing srtm_COL_ROW.tif
    tiles_dir = os.path.join(DATA_DIR, "srtm_tiles")
    os.makedirs(tiles_dir, exist_ok=True)

    # CGIAR tile grid: each tile is 5x5 degrees
    # col = floor((lon + 180) / 5) + 1
    # row = floor((60 - lat) / 5) + 1
    minx, miny, maxx, maxy = country_bounds
    if maxy > 60 or miny < -56:
        print("Warning: SRTM coverage is limited (~60N to ~56S). Results may be incomplete.")
    col_min = int((minx + 180) // 5) + 1
    col_max = int((maxx + 180) // 5) + 1
    row_min = int((60 - maxy) // 5) + 1
    row_max = int((60 - miny) // 5) + 1

    tile_coords = [
        (col, row)
        for col in range(col_min, col_max + 1)
        for row in range(row_min, row_max + 1)
    ]

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


def clip_dem_to_country(dem_path, country_gdf, country_slug):
    """Clip DEM to country boundary."""
    clipped_file = os.path.join(DATA_DIR, f"{country_slug}_dem_clipped.tif")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download SRTM and clip to a country boundary.")
    parser.add_argument("--country", default="Germany", help="Country name from Natural Earth (e.g. France).")
    args = parser.parse_args()

    country_name = args.country
    country_slug = slugify(country_name)

    print("=== Downloading Data for Sea Level Rise Visualization ===\n")
    print(f"Country: {country_name}\n")

    # Step 1: Country boundary
    country = download_country_boundary(country_name, country_slug)
    print(f"{country_name} bounds: {country.total_bounds}\n")

    # Step 2: SRTM elevation data
    dem_path = download_srtm_via_elevation_api(country.total_bounds, country_slug)

    # Step 3: Clip to country
    clipped_path = clip_dem_to_country(dem_path, country, country_slug)

    print("\n=== Data download complete! ===")
    print(f"Clipped DEM: {clipped_path}")
