"""
Sea Level Rise Visualization for a country
GPU-accelerated (PyTorch CUDA) rendering with all water bodies.
Pipes frames directly to ffmpeg - no PNG files needed.
"""

import os
import sys
import time
import re
import argparse
import urllib.request
import numpy as np
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
from shapely.geometry import box
from PIL import Image, ImageDraw, ImageFont
import subprocess
from scipy.ndimage import binary_dilation as scipy_binary_dilation
import torch
import torch.nn.functional as F

# === Configuration ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
DEFAULT_COUNTRY = "Germany"

# Video settings
WIDTH = 1920
HEIGHT = 1080
FPS = 60

# Sea level settings
SEA_LEVEL_MIN = 0
SEA_LEVEL_MAX = 1000
SEA_LEVEL_STEP = 1  # meters per frame

# Color scheme
OCEAN_COLOR_DEEP = [40, 80, 160]
OCEAN_COLOR_SHORE = [100, 160, 220]
RIVER_COLOR = [60, 120, 200]

# Terrain colormap
TERRAIN_COLORS = [
    (0,    (30, 120, 50)),
    (50,   (60, 160, 60)),
    (100,  (100, 180, 70)),
    (200,  (160, 200, 80)),
    (300,  (200, 210, 100)),
    (400,  (220, 200, 80)),
    (500,  (220, 170, 60)),
    (600,  (210, 140, 50)),
    (800,  (190, 100, 40)),
    (1000, (170, 70, 30)),
    (1500, (140, 50, 25)),
    (2000, (120, 100, 90)),
    (3000, (200, 200, 210)),
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hillshade smoothing (odd kernel size; set to 1 to disable)
HILLSHADE_SMOOTH_K = 3


def slugify(name):
    value = name.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "country"


def create_terrain_lut():
    """Pre-compute terrain color lookup table."""
    lut = np.zeros((3001, 3), dtype=np.uint8)
    for i in range(3001):
        if i <= TERRAIN_COLORS[0][0]:
            lut[i] = TERRAIN_COLORS[0][1]
            continue
        if i >= TERRAIN_COLORS[-1][0]:
            lut[i] = TERRAIN_COLORS[-1][1]
            continue
        for j in range(len(TERRAIN_COLORS) - 1):
            v0, c0 = TERRAIN_COLORS[j]
            v1, c1 = TERRAIN_COLORS[j + 1]
            if v0 <= i <= v1:
                t = (i - v0) / (v1 - v0)
                lut[i] = [int(c0[k] * (1 - t) + c1[k] * t) for k in range(3)]
                break
    return torch.from_numpy(lut).to(DEVICE)


def load_dem_data(dem_path):
    """Load the clipped DEM."""
    if not os.path.exists(dem_path):
        print(f"ERROR: DEM data not found: {dem_path}")
        print("Run download_data.py --country \"<Name>\" first, or pass --dem.")
        sys.exit(1)
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32)
        nodata = src.nodata
        bounds = src.bounds
    if nodata is not None:
        dem[dem == nodata] = np.nan
    dem[dem < -500] = np.nan
    print(f"DEM loaded: {dem.shape}, range: {np.nanmin(dem):.0f}m to {np.nanmax(dem):.0f}m")
    return dem, bounds


def download_if_missing(filename, url):
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        print(f"  Downloading {filename}...")
        urllib.request.urlretrieve(url, filepath)
    return filepath


def load_water_bodies(dem_path, water_tag):
    """Load ALL water bodies."""
    water_raster_path = os.path.join(DATA_DIR, f"water_bodies_{water_tag}.npy")
    if os.path.exists(water_raster_path):
        print("Water body raster already exists, loading...")
        return np.load(water_raster_path)

    with rasterio.open(dem_path) as src:
        dem_transform = src.transform
        dem_shape = (src.height, src.width)
        dem_bounds = src.bounds

    clip_box = box(dem_bounds.left, dem_bounds.bottom, dem_bounds.right, dem_bounds.top)

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

    print("Loading all water body datasets...")
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

            if geom_type == "line" and 'scalerank' in clipped.columns:
                for width, max_rank in [(3, 3), (2, 6), (1, 12)]:
                    subset = clipped[clipped['scalerank'] <= max_rank]
                    if len(subset) == 0:
                        continue
                    shapes = [(g, 1) for g in subset.geometry if g is not None]
                    layer = rasterize(shapes, out_shape=dem_shape, transform=dem_transform,
                                      fill=0, dtype=np.uint8, all_touched=True)
                    if width > 1:
                        layer = scipy_binary_dilation(layer, iterations=width - 1).astype(np.uint8)
                    water_raster = np.maximum(water_raster, layer)
            else:
                shapes = [(g, 1) for g in clipped.geometry if g is not None]
                layer = rasterize(shapes, out_shape=dem_shape, transform=dem_transform,
                                  fill=0, dtype=np.uint8, all_touched=True)
                water_raster = np.maximum(water_raster, layer)
        except Exception as e:
            print(f"  {name}: ERROR - {e}")

    np.save(water_raster_path, water_raster)
    print(f"  Water body raster saved ({np.sum(water_raster > 0)} water pixels)")
    return water_raster


def compute_hillshade_gpu(dem_np):
    """Pre-compute hillshade on GPU."""
    dem_filled = np.where(np.isnan(dem_np), 0, dem_np)
    d = torch.from_numpy(dem_filled).to(DEVICE, dtype=torch.float32)

    dy = torch.zeros_like(d)
    dx = torch.zeros_like(d)
    dy[1:-1, :] = (d[2:, :] - d[:-2, :]) / 2.0
    dx[:, 1:-1] = (d[:, 2:] - d[:, :-2]) / 2.0

    azimuth = 315.0 * 3.14159265 / 180.0
    altitude = 45.0 * 3.14159265 / 180.0

    slope = torch.atan(torch.sqrt(dx**2 + dy**2) * 3.0)
    aspect = torch.atan2(-dy, dx)

    hillshade = (torch.sin(torch.tensor(altitude, device=DEVICE)) * torch.cos(slope) +
                 torch.cos(torch.tensor(altitude, device=DEVICE)) * torch.sin(slope) *
                 torch.cos(torch.tensor(azimuth, device=DEVICE) - aspect))
    hillshade = torch.clamp(hillshade, 0.3, 1.0)
    if HILLSHADE_SMOOTH_K > 1 and HILLSHADE_SMOOTH_K % 2 == 1:
        hs = hillshade.unsqueeze(0).unsqueeze(0)
        hillshade = F.avg_pool2d(hs, kernel_size=HILLSHADE_SMOOTH_K,
                                 stride=1, padding=HILLSHADE_SMOOTH_K // 2).squeeze()
    return hillshade


def load_fonts():
    """Load fonts once."""
    font_paths = [
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/calibrib.ttf",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                return (ImageFont.truetype(fp, 72),
                        ImageFont.truetype(fp, 32),
                        ImageFont.truetype(fp, 18))
            except Exception:
                continue
    f = ImageFont.load_default()
    return f, f, f


def draw_text_on_frame(frame_img, sea_level, fonts):
    """Draw text overlay directly on an RGB PIL Image (no RGBA overhead)."""
    font_large, font_small, small_font = fonts
    draw = ImageDraw.Draw(frame_img)

    text = f"{int(sea_level)} m"
    x, y = 50, 30
    draw.text((x + 3, y + 3), text, font=font_large, fill=(0, 0, 0))
    draw.text((x, y), text, font=font_large, fill=(255, 255, 255))

    label = "Sea Level"
    draw.text((x + 3, y + 82), label, font=font_small, fill=(0, 0, 0))
    draw.text((x, y + 80), label, font=font_small, fill=(220, 220, 220))

    bar_y = HEIGHT - 40
    bar_h = 8
    bar_margin = 50
    bar_width = WIDTH - 2 * bar_margin
    progress = sea_level / SEA_LEVEL_MAX

    draw.rectangle([bar_margin, bar_y, bar_margin + bar_width, bar_y + bar_h], fill=(40, 40, 40))
    fill_width = int(bar_width * progress)
    if fill_width > 0:
        draw.rectangle([bar_margin, bar_y, bar_margin + fill_width, bar_y + bar_h], fill=(100, 180, 255))

    tick_step = 250
    for lv in range(0, SEA_LEVEL_MAX + 1, tick_step):
        lx = bar_margin + int(bar_width * (lv / SEA_LEVEL_MAX))
        draw.line([(lx, bar_y - 4), (lx, bar_y + bar_h + 4)], fill=(150, 150, 150), width=1)
        lt = f"{lv}m"
        bbox = draw.textbbox((0, 0), lt, font=small_font)
        tw = bbox[2] - bbox[0]
        draw.text((lx - tw // 2, bar_y + bar_h + 6), lt, font=small_font, fill=(180, 180, 180))


def precompute_scaled_dims(dem_shape):
    """Pre-compute the resize dimensions and offsets for fit_to_frame on GPU."""
    h, w = dem_shape
    scale = min(WIDTH / w, HEIGHT / h) * 0.88
    new_w = int(w * scale)
    new_h = int(h * scale)
    offset_x = (WIDTH - new_w) // 2
    offset_y = (HEIGHT - new_h) // 2
    return new_w, new_h, offset_x, offset_y


@torch.no_grad()
def render_frame_gpu(dem_gpu, is_nodata_gpu, sea_level, terrain_lut_gpu,
                     water_mask_gpu, hillshade_gpu, ocean_noise_gpu,
                     shore_t, deep_t, river_t, deep_bg_t):
    """Render a single frame entirely on GPU. Returns HxWx3 float tensor on GPU."""
    # Masks
    is_land = (~is_nodata_gpu) & (dem_gpu >= sea_level)
    is_flooded = (~is_nodata_gpu) & (dem_gpu < sea_level)

    # Output image
    img = torch.zeros_like(hillshade_gpu).unsqueeze(2).expand(-1, -1, 3).contiguous()
    img = img * 0  # zero it out, keep shape

    # Terrain coloring via LUT
    land_elev = torch.clamp(dem_gpu - sea_level, 0, 3000).to(torch.int64)
    land_elev[~is_land] = 0
    img[is_land] = terrain_lut_gpu[land_elev[is_land]].float()

    # Flooded areas
    if torch.any(is_flooded):
        max_vis_depth = max(50.0, sea_level * 0.5)
        water_depth = sea_level - dem_gpu[is_flooded]
        depth_norm = torch.clamp(water_depth / max_vis_depth, 0.0, 1.0).unsqueeze(1)
        img[is_flooded] = shore_t * (1 - depth_norm) + deep_t * depth_norm

    # Water bodies on land
    is_water_on_land = water_mask_gpu & is_land
    if torch.any(is_water_on_land):
        img[is_water_on_land] = river_t

    # Background ocean
    img[is_nodata_gpu] = deep_bg_t

    # Hillshade on land
    hs = hillshade_gpu.unsqueeze(2)
    land_3d = is_land.unsqueeze(2)
    img = torch.where(land_3d, img * hs, img)

    # Ocean noise
    ocean_3d = (is_nodata_gpu | is_flooded).unsqueeze(2)
    noise_3d = ocean_noise_gpu.unsqueeze(2)
    img = torch.where(ocean_3d, img * noise_3d, img)

    # Coastline glow via max_pool
    water_float = (is_flooded | is_nodata_gpu).float().unsqueeze(0).unsqueeze(0)
    dilated = F.max_pool2d(water_float, kernel_size=5, stride=1, padding=2).squeeze() > 0
    coastline = dilated & is_land
    if torch.any(coastline):
        img[coastline] = img[coastline] * 0.7 + 255.0 * 0.3

    return torch.clamp(img, 0, 255)


def parse_args():
    parser = argparse.ArgumentParser(description="Sea level rise visualization renderer.")
    parser.add_argument("--country", default=DEFAULT_COUNTRY,
                        help="Country name (must match Natural Earth, e.g. Germany).")
    parser.add_argument("--dem", default=None, help="Path to clipped DEM .tif (overrides --country).")
    parser.add_argument("--output", default=None, help="Output video path (.mp4).")
    return parser.parse_args()


def main():
    args = parse_args()
    country_name = args.country
    country_slug = slugify(country_name)
    dem_path = args.dem or os.path.join(DATA_DIR, f"{country_slug}_dem_clipped.tif")
    dem_tag = country_slug
    if args.dem:
        dem_base = os.path.splitext(os.path.basename(args.dem))[0]
        dem_tag = slugify(dem_base.replace("_dem_clipped", ""))
    output_video = args.output or os.path.join(SCRIPT_DIR, f"sea_level_rise_{dem_tag}.mp4")

    print(f"=== Sea Level Rise Visualization - {country_name} (GPU) ===\n")
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {mem_gb:.1f} GB")
    print(f"DEM: {dem_path}")
    print(f"Output: {output_video}")
    print()

    # Load data
    print("Loading DEM data...")
    dem_np, bounds = load_dem_data(dem_path)

    print("Loading water bodies...")
    water_np = load_water_bodies(dem_path, dem_tag)

    print("Creating terrain colormap...")
    terrain_lut_gpu = create_terrain_lut()

    # Move data to GPU
    print("Uploading data to GPU...")
    is_nodata_np = np.isnan(dem_np)
    dem_clean = np.where(is_nodata_np, 0, dem_np)

    dem_gpu = torch.from_numpy(dem_clean).to(DEVICE, dtype=torch.float32)
    is_nodata_gpu = torch.from_numpy(is_nodata_np).to(DEVICE, dtype=torch.bool)
    water_mask_gpu = torch.from_numpy(water_np > 0).to(DEVICE, dtype=torch.bool)

    # Pre-allocate color tensors on GPU (avoid re-creating each frame)
    shore_t = torch.tensor(OCEAN_COLOR_SHORE, dtype=torch.float32, device=DEVICE)
    deep_t = torch.tensor(OCEAN_COLOR_DEEP, dtype=torch.float32, device=DEVICE)
    river_t = torch.tensor(RIVER_COLOR, dtype=torch.float32, device=DEVICE)
    deep_bg_t = torch.tensor(OCEAN_COLOR_DEEP, dtype=torch.float32, device=DEVICE)

    print("Computing hillshade on GPU...")
    hillshade_gpu = compute_hillshade_gpu(dem_np)

    print("Generating ocean noise...")
    from scipy.ndimage import gaussian_filter
    noise_np = np.random.RandomState(42).uniform(0.95, 1.05, dem_np.shape).astype(np.float32)
    noise_np = gaussian_filter(noise_np, sigma=5)
    ocean_noise_gpu = torch.from_numpy(noise_np).to(DEVICE, dtype=torch.float32)

    # Sea levels
    sea_levels = np.arange(SEA_LEVEL_MIN, SEA_LEVEL_MAX + SEA_LEVEL_STEP, SEA_LEVEL_STEP)
    total_frames = len(sea_levels)
    duration_sec = total_frames / FPS
    print(f"\nTotal frames: {total_frames} (every {SEA_LEVEL_STEP}m)")
    print(f"Video duration: {duration_sec:.1f}s at {FPS}fps")

    # Load fonts
    fonts = load_fonts()

    # Pre-compute resize dimensions
    new_w, new_h, off_x, off_y = precompute_scaled_dims(dem_np.shape)

    # Pre-render background frame
    bg_frame = Image.new('RGB', (WIDTH, HEIGHT), tuple(OCEAN_COLOR_DEEP))
    bg_array = np.array(bg_frame)

    # Start ffmpeg pipe - write raw RGB frames directly
    print("\nStarting ffmpeg pipe...")
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{WIDTH}x{HEIGHT}",
        "-pix_fmt", "rgb24",
        "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_video
    ]
    ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print("Generating frames...\n")
    t_start = time.time()

    for i, sea_level in enumerate(sea_levels):
        sl = float(sea_level)

        # Render on GPU
        img_gpu = render_frame_gpu(
            dem_gpu, is_nodata_gpu, sl, terrain_lut_gpu,
            water_mask_gpu, hillshade_gpu, ocean_noise_gpu,
            shore_t, deep_t, river_t, deep_bg_t
        )

        # GPU resize: (H,W,3) -> (1,3,H,W) for interpolate, then back
        img_chw = img_gpu.permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
        img_resized = F.interpolate(img_chw, size=(new_h, new_w), mode='bilinear', align_corners=False)
        img_resized = img_resized.squeeze(0).permute(1, 2, 0)  # (new_h, new_w, 3)
        img_small = img_resized.to(torch.uint8).cpu().numpy()

        # Compose into 1920x1080 frame
        frame_array = bg_array.copy()
        frame_array[off_y:off_y + new_h, off_x:off_x + new_w] = img_small

        # Draw text overlay directly
        frame_img = Image.fromarray(frame_array)
        draw_text_on_frame(frame_img, sl, fonts)

        # Write raw RGB bytes to ffmpeg
        ffmpeg_proc.stdin.write(frame_img.tobytes())

        if (i + 1) % 60 == 0 or i == total_frames - 1:
            elapsed = time.time() - t_start
            fps_actual = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total_frames - i - 1) / fps_actual if fps_actual > 0 else 0
            pct = (i + 1) / total_frames * 100
            print(f"  Frame {i + 1}/{total_frames} ({pct:.0f}%) - {sl:.0f}m - "
                  f"{fps_actual:.1f} fps - ETA {eta:.0f}s")

    # Close ffmpeg
    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()

    elapsed_total = time.time() - t_start
    print(f"\nAll frames generated in {elapsed_total:.1f}s ({total_frames / elapsed_total:.1f} fps)")

    if ffmpeg_proc.returncode == 0:
        size_mb = os.path.getsize(output_video) / (1024 * 1024)
        print(f"\nVideo saved to: {output_video}")
        print(f"File size: {size_mb:.1f} MB")
    else:
        print(f"FFmpeg failed with return code {ffmpeg_proc.returncode}")
        sys.exit(1)

    print("\n=== Done! ===")


if __name__ == "__main__":
    main()
