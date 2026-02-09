"""
Download ETOPO 2022 15s ice surface elevation GeoTIFF tiles
and merge them into a single world GeoTIFF.

This is a large download. Recommended to use a drive with lots of free space.
"""

import argparse
import os
import re
import shutil
import time
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import rasterio
from rasterio.merge import merge

ETOPO_DIR_URL = (
    "https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2022/data/15s/"
    "15s_surface_elev_gtif/"
)


def list_tiles():
    html = urllib.request.urlopen(ETOPO_DIR_URL, timeout=60).read().decode(
        "utf-8", "ignore"
    )
    names = re.findall(
        r'href=[\'"]([^\'"]+\.tif)[\'"]', html, flags=re.IGNORECASE
    )
    return sorted({n for n in names if n.lower().endswith(".tif")})


def download_tiles(tile_names, out_dir):
    raise RuntimeError("download_tiles() should not be used directly.")


def download_one(name, out_dir, timeout, retries):
    dest = out_dir / name
    if dest.exists():
        return ("skip", name)
    url = f"{ETOPO_DIR_URL}{name}"
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                with open(dest, "wb") as f:
                    shutil.copyfileobj(resp, f, length=1024 * 1024)
            return ("ok", name)
        except Exception as e:
            if dest.exists():
                try:
                    dest.unlink()
                except Exception:
                    pass
            if attempt >= retries:
                return ("fail", name, str(e))
            time.sleep(1.5 * (attempt + 1))


def download_tiles_parallel(tile_names, out_dir, workers, timeout, retries):
    out_dir.mkdir(parents=True, exist_ok=True)
    total = len(tile_names)
    done = 0
    failed = 0
    skipped = 0
    print(f"Downloading {total} tiles with {workers} workers...")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(download_one, name, out_dir, timeout, retries)
            for name in tile_names
        ]
        for fut in as_completed(futures):
            res = fut.result()
            status = res[0]
            done += 1
            if status == "fail":
                failed += 1
                print(f"FAILED: {res[1]} -> {res[2]}")
            elif status == "skip":
                skipped += 1
            if done % 20 == 0 or done == total:
                print(f"  {done}/{total} complete (skipped {skipped}, failed {failed})")
    if failed:
        raise RuntimeError(f"{failed} downloads failed. Re-run to retry.")


def merge_tiles(tile_paths, output_path, mem_limit):
    print("Merging tiles (this can take a while)...")
    datasets = [rasterio.open(p) for p in tile_paths]
    profile = datasets[0].profile.copy()
    profile.update(
        driver="GTiff",
        compress="lzw",
        tiled=True,
        blockxsize=512,
        blockysize=512,
        bigtiff="yes",
        nodata=-9999,
    )
    merge(
        datasets,
        dst_path=str(output_path),
        nodata=-9999,
        mem_limit=mem_limit,
        dst_kwds=profile,
    )
    for ds in datasets:
        ds.close()
    print(f"World mosaic written to: {output_path}")


def default_out_dir():
    return Path(__file__).resolve().parent / "data" / "world"


def main():
    parser = argparse.ArgumentParser(
        description="Download and merge ETOPO 2022 15s ice surface elevation tiles."
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory to store tiles (default: data/world).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output GeoTIFF path (default: <out-dir>/etopo2022_surface_15s_world.tif).",
    )
    parser.add_argument(
        "--mem-limit",
        type=int,
        default=1024,
        help="Merge memory limit in MB (default: 1024).",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Only download tiles, do not merge.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Parallel download workers (default: 6).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Per-tile download timeout in seconds (default: 60).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Retries per tile on failure (default: 2).",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir()
    output_path = Path(args.output) if args.output else out_dir / "etopo2022_surface_15s_world.tif"

    print(f"Listing tiles from {ETOPO_DIR_URL} ...")
    tiles = list_tiles()
    print(f"Found {len(tiles)} tiles.")
    if not tiles:
        raise RuntimeError(
            "No tiles found. The NOAA index page format may have changed."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading to: {out_dir}")
    download_tiles_parallel(tiles, out_dir, args.workers, args.timeout, args.retries)

    if args.no_merge:
        print("Download complete. Skipping merge (--no-merge).")
        return

    tile_paths = [out_dir / name for name in tiles]
    merge_tiles(tile_paths, output_path, args.mem_limit)


if __name__ == "__main__":
    main()
