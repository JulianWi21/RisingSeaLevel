"""
Download elevation data (SRTM) and Germany boundary for sea level rise visualization.
Uses NASA SRTM data via OpenTopography-style direct tile downloads and Natural Earth boundaries.
"""

import os
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


def download_germany_boundary():
    """Download Germany boundary from Natural Earth 10m countries."""
    boundary_file = os.path.join(DATA_DIR, "germany_boundary.gpkg")
    if os.path.exists(boundary_file):
        print("Germany boundary already exists, loading...")
        return gpd.read_file(boundary_file)

    print("Downloading Natural Earth 10m countries...")
    url = "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip"
    zip_path = os.path.join(DATA_DIR, "ne_10m_admin_0_countries.zip")

    urllib.request.urlretrieve(url, zip_path)
    print("Download complete. Extracting Germany...")

    world = gpd.read_file(f"zip://{zip_path}")
    germany = world[world['NAME'] == 'Germany'].copy()
    germany = germany.to_crs("EPSG:4326")
    germany.to_file(boundary_file, driver="GPKG")

    # Cleanup
    os.remove(zip_path)
    print(f"Germany boundary saved to {boundary_file}")
    return germany


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


def download_srtm_via_elevation_api():
    """
    Download SRTM data for Germany using a direct approach.
    Downloads from CGIAR SRTM 90m (5x5 degree tiles).
    """
    merged_file = os.path.join(DATA_DIR, "germany_srtm.tif")
    if os.path.exists(merged_file):
        print("SRTM data already exists.")
        return merged_file

    # CGIAR SRTM v4.1 tiles covering Germany
    # Tile numbering: col 38-39, row 01-03 cover Europe including Germany
    # Format: srtm_COL_ROW.zip containing srtm_COL_ROW.tif
    tiles_dir = os.path.join(DATA_DIR, "srtm_tiles")
    os.makedirs(tiles_dir, exist_ok=True)

    # Germany: ~5-16E, ~47-55.5N
    # CGIAR tile grid: each tile is 5x5 degrees
    # Starting from -180, -60: col = (lon + 180) / 5 + 1, row = (60 - lat) / 5 + 1
    # For lon 5: col = 38, lon 10: col = 39, lon 15: col = 40
    # For lat 55: row = 2, lat 50: row = 3, lat 45: row = 4

    tile_coords = [
        (38, 2), (39, 2), (40, 2),  # Northern part
        (38, 3), (39, 3), (40, 3),  # Central part
        (38, 4), (39, 4), (40, 4),  # Southern part (Alps)
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


def clip_dem_to_germany(dem_path, germany_gdf):
    """Clip DEM to Germany boundary."""
    clipped_file = os.path.join(DATA_DIR, "germany_dem_clipped.tif")
    if os.path.exists(clipped_file):
        print("Clipped DEM already exists.")
        return clipped_file

    print("Clipping DEM to Germany boundary...")
    germany_4326 = germany_gdf.to_crs("EPSG:4326")

    with rasterio.open(dem_path) as src:
        out_image, out_transform = rasterio_mask(
            src,
            germany_4326.geometry,
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
    print("=== Downloading Data for Sea Level Rise Visualization ===\n")

    # Step 1: Germany boundary
    germany = download_germany_boundary()
    print(f"Germany bounds: {germany.total_bounds}\n")

    # Step 2: SRTM elevation data
    dem_path = download_srtm_via_elevation_api()

    # Step 3: Clip to Germany
    clipped_path = clip_dem_to_germany(dem_path, germany)

    print("\n=== Data download complete! ===")
    print(f"Clipped DEM: {clipped_path}")
