"""
Clip the ETOPO world GeoTIFF to a specific country boundary,
then optionally resample to a target size.
"""
import os
import sys
import argparse
import re
import urllib.request
import numpy as np
import rasterio
from rasterio.mask import mask as rasterio_mask
from rasterio.warp import Resampling
from affine import Affine
import geopandas as gpd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")


def slugify(name):
    value = name.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "country"


def get_country_boundary(country_name, country_slug, country_dir):
    boundary_file = os.path.join(country_dir, f"{country_slug}_boundary.gpkg")
    if os.path.exists(boundary_file):
        print(f"Boundary already exists: {boundary_file}")
        return gpd.read_file(boundary_file)

    print("Downloading Natural Earth 10m countries...")
    url = "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip"
    zip_path = os.path.join(country_dir, "ne_10m_admin_0_countries.zip")
    urllib.request.urlretrieve(url, zip_path)

    world = gpd.read_file(f"zip://{zip_path}")
    name_lower = country_name.strip().lower()
    for col in ["NAME", "NAME_EN", "ADMIN", "SOVEREIGNT"]:
        if col in world.columns:
            matches = world[world[col].fillna("").str.lower() == name_lower]
            if not matches.empty:
                country = matches.copy()
                break
    else:
        raise ValueError(f"Country '{country_name}' not found.")

    if len(country) > 1:
        country = gpd.GeoDataFrame(
            {"geometry": [country.geometry.unary_union]}, crs=country.crs
        )
    country = country.to_crs("EPSG:4326")
    country.to_file(boundary_file, driver="GPKG")
    os.remove(zip_path)
    print(f"Boundary saved: {boundary_file}")
    return country


def clip_etopo(etopo_path, country_gdf, output_path):
    if os.path.exists(output_path):
        print(f"Clipped DEM already exists: {output_path}")
        return output_path

    print(f"Clipping ETOPO to country boundary...")
    country_4326 = country_gdf.to_crs("EPSG:4326")

    with rasterio.open(etopo_path) as src:
        out_image, out_transform = rasterio_mask(
            src, country_4326.geometry, crop=True, nodata=-9999, filled=True
        )
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "nodata": -9999,
            "compress": "lzw",
        })

    with rasterio.open(output_path, "w", **out_meta) as dst:
        dst.write(out_image)

    print(f"Clipped DEM saved: {output_path} ({out_image.shape[2]}x{out_image.shape[1]})")
    return output_path


def resample_dem(clipped_path, output_path, target_w, target_h):
    if os.path.exists(output_path):
        print(f"Resized DEM already exists: {output_path}")
        return output_path

    with rasterio.open(clipped_path) as src:
        src_w, src_h = src.width, src.height
        scale = min(target_w / src_w, target_h / src_h, 1.0)
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))

        print(f"Resampling {src_w}x{src_h} -> {new_w}x{new_h}...")
        data = src.read(
            1,
            out_shape=(new_h, new_w),
            resampling=Resampling.average if (new_w < src_w) else Resampling.bilinear,
            masked=True,
        )
        scale_x = src_w / new_w
        scale_y = src_h / new_h
        transform = src.transform * Affine.scale(scale_x, scale_y)

        profile = src.profile.copy()
        profile.update({
            "height": new_h,
            "width": new_w,
            "transform": transform,
            "compress": "lzw",
        })
        filled = data.filled(src.nodata if src.nodata is not None else -9999)
        if profile.get("nodata") is None:
            profile["nodata"] = -9999

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(filled, 1)

    print(f"Resized DEM saved: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Clip ETOPO world DEM to a country.")
    parser.add_argument("country", help="Country name (e.g. Austria)")
    parser.add_argument("--etopo", default=os.path.join(DATA_DIR, "world", "etopo2022_surface_15s_world.tif"))
    parser.add_argument("--dem-size", default=None, help="Target resize e.g. '340x260'")
    args = parser.parse_args()

    country_slug = slugify(args.country)
    country_dir = os.path.join(DATA_DIR, country_slug)
    os.makedirs(country_dir, exist_ok=True)

    country_gdf = get_country_boundary(args.country, country_slug, country_dir)
    clipped_path = os.path.join(country_dir, f"{country_slug}_etopo_dem_clipped.tif")
    clip_etopo(args.etopo, country_gdf, clipped_path)

    if args.dem_size:
        m = re.match(r"(\d+)x(\d+)", args.dem_size)
        if m:
            tw, th = int(m.group(1)), int(m.group(2))
            with rasterio.open(clipped_path) as src:
                src_w, src_h = src.width, src.height
                scale = min(tw / src_w, th / src_h, 1.0)
                new_w = max(1, int(round(src_w * scale)))
                new_h = max(1, int(round(src_h * scale)))
            resized_path = os.path.join(
                country_dir, f"{country_slug}_{new_w}x{new_h}_etopo_dem_clipped.tif"
            )
            resample_dem(clipped_path, resized_path, tw, th)

    print("\nDone!")


if __name__ == "__main__":
    main()
