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
WIDTH = 3840
HEIGHT = 2160
FPS = 60
RESIZED_MAX_SCALE = 2.0  # default: pick resized DEM up to 2x video size

# UI scale relative to 1080p baseline
UI_SCALE = min(WIDTH / 1920, HEIGHT / 1080)

# Sea level settings
SEA_LEVEL_MIN = 0
SEA_LEVEL_MAX = 2000
SEA_LEVEL_STEP = 0.25  # meters per frame
MONT_BLANC_ELEVATION_M = 4809
GROSSGLOCKNER_ELEVATION_M = 3798
FADE_DURATION_SEC = 2.0  # fade-to-black at start/end of video
HOLD_START_SEC = 1.0     # hold at sea_min after fade-in
HOLD_END_SEC = 5.0       # hold at sea_max before fade-out

# Color scheme
OCEAN_COLOR_DEEP = [40, 80, 160]
OCEAN_COLOR_SHORE = [100, 160, 220]
RIVER_COLOR = [40, 80, 160]
LAKE_SHORE_PX = 20          # rim width in pixels (at SRTM 90m ref resolution)

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

# Land appearance tuning
LAND_BRIGHTNESS = 2.00  # 1.0 = unchanged
LAND_SATURATION = 1.20  # 1.0 = unchanged
LAND_BRIGHTNESS_ETOPO = 1.45
LAND_SATURATION_ETOPO = 1.10
HILLSHADE_STRENGTH_ETOPO = 0.65
HILLSHADE_GAIN_ETOPO = 0.92
HILLSHADE_STRENGTH_RESIZED = 0.85  # reduce hillshade contrast for resized DEMs
HILLSHADE_BLUR_ITERS_RESIZED = 2  # softening for resized DEMs (0 = off)
HILLSHADE_GAIN_RESIZED = 0.87     # overall darkening for resized hillshade
# Supersampling factor for resize (1.0 = off, higher = smoother, slower)
SUPERSAMPLE = 1.0

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
        print("Run download_data.py <Name> (or --country \"<Name>\") first, or pass --dem.")
        sys.exit(1)
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32)
        nodata = src.nodata
        bounds = src.bounds
    if nodata is not None:
        dem[dem == nodata] = np.nan
    dem[dem < -500] = np.nan
    dem_min = float(np.nanmin(dem))
    dem_max = float(np.nanmax(dem))
    print(f"DEM loaded: {dem.shape}, range: {dem_min:.0f}m to {dem_max:.0f}m")
    return dem, bounds, dem_min, dem_max


def download_if_missing(filename, url):
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        print(f"  Downloading {filename}...")
        urllib.request.urlretrieve(url, filepath)
    return filepath


def load_water_bodies(dem_path, water_tag, data_dir):
    """Load ALL water bodies (separate rivers and lakes)."""
    water_rivers_path = os.path.join(data_dir, f"water_rivers_{water_tag}.npy")
    water_lakes_path = os.path.join(data_dir, f"water_lakes_{water_tag}.npy")
    with rasterio.open(dem_path) as src:
        dem_transform = src.transform
        dem_shape = (src.height, src.width)
        dem_bounds = src.bounds
    if os.path.exists(water_rivers_path) and os.path.exists(water_lakes_path):
        print("Water body rasters already exist, loading...")
        rivers_existing = np.load(water_rivers_path)
        lakes_existing = np.load(water_lakes_path)
        if rivers_existing.shape == dem_shape and lakes_existing.shape == dem_shape:
            return rivers_existing, lakes_existing
        print(
            f"Water body raster shape mismatch (rivers {rivers_existing.shape}, lakes {lakes_existing.shape} vs {dem_shape}), rebuilding..."
        )
        for path in (water_rivers_path, water_lakes_path):
            try:
                os.remove(path)
            except OSError:
                pass

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

    # Scale dilation iterations by DEM resolution (reference: SRTM ~90m/pixel)
    pixel_deg = abs(dem_transform.a)
    pixel_m = pixel_deg * 111000
    ref_pixel_m = 90.0
    dilation_scale = ref_pixel_m / pixel_m
    print(f"Loading all water body datasets (pixel ~{pixel_m:.0f}m, dilation scale {dilation_scale:.2f})...")
    rivers_raster = np.zeros(dem_shape, dtype=np.uint8)
    lakes_raster = np.zeros(dem_shape, dtype=np.uint8)

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
                    scaled_iters = max(0, round((width - 1) * dilation_scale))
                    if scaled_iters > 0:
                        layer = scipy_binary_dilation(layer, iterations=scaled_iters).astype(np.uint8)
                    rivers_raster = np.maximum(rivers_raster, layer)
            else:
                shapes = [(g, 1) for g in clipped.geometry if g is not None]
                layer = rasterize(shapes, out_shape=dem_shape, transform=dem_transform,
                                  fill=0, dtype=np.uint8, all_touched=True)
                lakes_raster = np.maximum(lakes_raster, layer)
        except Exception as e:
            print(f"  {name}: ERROR - {e}")

    np.save(water_rivers_path, rivers_raster)
    np.save(water_lakes_path, lakes_raster)
    print(
        f"  Water rasters saved (rivers {np.sum(rivers_raster > 0)} px, lakes {np.sum(lakes_raster > 0)} px)"
    )
    return rivers_raster, lakes_raster


def compute_hillshade_gpu(dem_np, pixel_size_m=90.0):
    """Pre-compute hillshade on GPU. Slope exaggeration scales with pixel size."""
    dem_filled = np.where(np.isnan(dem_np), 0, dem_np)
    d = torch.from_numpy(dem_filled).to(DEVICE, dtype=torch.float32)

    dy = torch.zeros_like(d)
    dx = torch.zeros_like(d)
    dy[1:-1, :] = (d[2:, :] - d[:-2, :]) / 2.0
    dx[:, 1:-1] = (d[:, 2:] - d[:, :-2]) / 2.0

    azimuth = 315.0 * 3.14159265 / 180.0
    altitude = 45.0 * 3.14159265 / 180.0

    exag = 3.0 * (90.0 / pixel_size_m)
    slope = torch.atan(torch.sqrt(dx**2 + dy**2) * exag)
    aspect = torch.atan2(-dy, dx)

    hillshade = (torch.sin(torch.tensor(altitude, device=DEVICE)) * torch.cos(slope) +
                 torch.cos(torch.tensor(altitude, device=DEVICE)) * torch.sin(slope) *
                 torch.cos(torch.tensor(azimuth, device=DEVICE) - aspect))
    return torch.clamp(hillshade, 0.3, 1.0)


def blur_hillshade_gpu(hillshade_gpu, iterations=1):
    """Lightly blur hillshade to reduce sharpness on resized DEMs."""
    if iterations <= 0:
        return hillshade_gpu
    hs = hillshade_gpu.unsqueeze(0).unsqueeze(0)
    for _ in range(iterations):
        hs = F.avg_pool2d(hs, kernel_size=3, stride=1, padding=1)
    return hs.squeeze(0).squeeze(0)


def build_lake_rim_weight(lakes_mask_gpu, rim_px):
    """Create a 0..1 weight map (1 at lake edge, 0 inside) with smooth falloff."""
    if rim_px <= 0:
        return None
    weights = torch.zeros_like(lakes_mask_gpu, dtype=torch.float32)
    eroded = lakes_mask_gpu
    for d in range(rim_px):
        inv = (~eroded).float().unsqueeze(0).unsqueeze(0)
        eroded_next = (F.max_pool2d(inv, kernel_size=3, stride=1, padding=1) == 0).squeeze()
        ring = eroded & ~eroded_next
        # Quadratic falloff: stays brighter (shore-like) longer, then drops off
        t = d / rim_px
        w = (1.0 - t) ** 2
        if torch.any(ring):
            weights[ring] = w
        eroded = eroded_next
        if not torch.any(eroded):
            break
    return weights



def load_fonts():
    """Load fonts once."""
    scale = UI_SCALE
    font_paths = [
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/calibrib.ttf",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                return (ImageFont.truetype(fp, int(72 * scale)),
                        ImageFont.truetype(fp, int(48 * scale)),
                        ImageFont.truetype(fp, int(18 * scale)))
            except Exception:
                continue
    f = ImageFont.load_default()
    return f, f, f


def draw_text_on_frame(frame_img, sea_level, fonts, sea_min, sea_max, ui_tick_step=None):
    """Draw text overlay directly on an RGB PIL Image (no RGBA overhead)."""
    font_large, font_small, small_font = fonts
    draw = ImageDraw.Draw(frame_img)
    s = UI_SCALE

    text = f"{int(sea_level)} m"
    x, y = int(50 * s), int(30 * s)
    shadow = max(1, int(3 * s))
    draw.text((x + shadow, y + shadow), text, font=font_large, fill=(0, 0, 0))
    draw.text((x, y), text, font=font_large, fill=(255, 255, 255))

    label = "Sea Level"
    draw.text((x + shadow, y + int(82 * s) + shadow), label, font=font_small, fill=(0, 0, 0))
    draw.text((x, y + int(80 * s)), label, font=font_small, fill=(220, 220, 220))


def load_cities(country_name, dem_path, num_cities=10):
    """Load top N cities for a country from Natural Earth populated places.
    Returns list of dicts with name, lon, lat, population, elevation (from DEM)."""
    places_file = os.path.join(DATA_DIR, "ne_10m_populated_places.zip")
    url = "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_populated_places.zip"
    if not os.path.exists(places_file):
        print(f"  Downloading populated places...")
        try:
            urllib.request.urlretrieve(url, places_file)
        except Exception:
            # Fallback: try downloading with PowerShell on Windows
            import platform
            if platform.system() == "Windows":
                subprocess.run(
                    ["powershell", "-Command",
                     f"Invoke-WebRequest -Uri '{url}' -OutFile '{places_file}'"],
                    check=True, capture_output=True
                )
            else:
                raise
    places = gpd.read_file(f"zip://{places_file}")
    places = places.to_crs("EPSG:4326")
    # Match country
    name_lower = country_name.strip().lower()
    matched = None
    for col in ["SOV0NAME", "ADM0NAME", "ADM0_A3"]:
        if col in places.columns:
            series = places[col].fillna("").astype(str).str.lower()
            m = places[series == name_lower]
            if not m.empty:
                matched = m
                break
    if matched is None or matched.empty:
        print(f"  Warning: No cities found for '{country_name}'.")
        return []
    # Sort by population
    pop_col = None
    for c in ["POP_MAX", "POP_MIN", "GN_POP"]:
        if c in matched.columns:
            pop_col = c
            break
    if pop_col:
        matched = matched.sort_values(pop_col, ascending=False)
    cities = matched.head(num_cities)
    # Get elevation from DEM for each city
    result = []
    with rasterio.open(dem_path) as src:
        dem_data = src.read(1)
        for _, row in cities.iterrows():
            lon, lat = row.geometry.x, row.geometry.y
            try:
                py, px = src.index(lon, lat)
                if 0 <= py < src.height and 0 <= px < src.width:
                    elev = float(dem_data[py, px])
                    if elev < -500 or (src.nodata is not None and dem_data[py, px] == src.nodata):
                        elev = 0.0
                else:
                    elev = 0.0
            except Exception:
                elev = 0.0
            name = row.get("NAME", row.get("NAME_EN", "?"))
            pop = int(row.get(pop_col, 0)) if pop_col else 0
            result.append({"name": name, "lon": lon, "lat": lat,
                           "population": pop, "elevation": elev})
    print(f"  Loaded {len(result)} cities: {', '.join(c['name'] for c in result)}")
    return result


def project_cities_to_frame(cities, dem_path, new_w, new_h, off_x, off_y):
    """Convert city lon/lat to video frame pixel coordinates."""
    if not cities:
        return []
    with rasterio.open(dem_path) as src:
        dem_h, dem_w = src.height, src.width
        scale = min(WIDTH / dem_w, HEIGHT / dem_h)
        for city in cities:
            py, px = src.index(city["lon"], city["lat"])
            # Scale to resized frame
            city["frame_x"] = int(px * scale) + off_x
            city["frame_y"] = int(py * scale) + off_y
    return cities


def resolve_city_label_positions(cities, font_city):
    """Pre-compute non-overlapping label positions for all cities."""
    if not cities or font_city is None:
        return
    s = UI_SCALE
    pad = int(3 * s)
    offset = int(8 * s)

    placed_boxes = []  # (x1, y1, x2, y2) of already-placed labels

    for city in cities:
        fx, fy = city["frame_x"], city["frame_y"]
        name = city["name"]
        bbox = font_city.getbbox(name)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        # 8 candidate offsets (dx, dy) relative to dot
        candidates = [
            ( offset,             -th // 2),            # right
            (-offset - tw,        -th // 2),            # left
            (-tw // 2,            -offset - th),        # above
            (-tw // 2,             offset),             # below
            ( offset,             -offset - th),        # upper-right
            ( offset,              offset),             # lower-right
            (-offset - tw,        -offset - th),        # upper-left
            (-offset - tw,         offset),             # lower-left
        ]

        best_pos = None
        best_overlap = float('inf')
        best_box = None

        for dx, dy in candidates:
            tx, ty = fx + dx, fy + dy
            box = (tx - pad, ty - pad, tx + tw + pad, ty + th + pad)
            # Prefer positions inside the frame
            if box[0] < 0 or box[1] < 0 or box[2] > WIDTH or box[3] > HEIGHT:
                continue
            overlap = 0
            for pb in placed_boxes:
                ix1 = max(box[0], pb[0])
                iy1 = max(box[1], pb[1])
                ix2 = min(box[2], pb[2])
                iy2 = min(box[3], pb[3])
                if ix1 < ix2 and iy1 < iy2:
                    overlap += (ix2 - ix1) * (iy2 - iy1)
            if overlap < best_overlap:
                best_overlap = overlap
                best_pos = (tx, ty)
                best_box = box
                if overlap == 0:
                    break

        if best_pos is None:
            # All candidates were outside the frame; fall back to right
            tx, ty = fx + offset, fy - th // 2
            best_pos = (tx, ty)
            best_box = (tx - pad, ty - pad, tx + tw + pad, ty + th + pad)

        city["label_x"] = best_pos[0]
        city["label_y"] = best_pos[1]
        placed_boxes.append(best_box)


def draw_cities_on_frame(draw, cities, sea_level, font_city):
    """Draw city markers and labels. Cities always stay visible."""
    s = UI_SCALE
    dot_r = max(2, int(4 * s))
    shadow = max(1, int(2 * s))
    for city in cities:
        fx, fy = city["frame_x"], city["frame_y"]
        name = city["name"]
        # Skip if outside frame
        if fx < 0 or fx >= WIDTH or fy < 0 or fy >= HEIGHT:
            continue
        white = (255, 255, 255)
        # Draw dot
        draw.ellipse([fx - dot_r, fy - dot_r, fx + dot_r, fy + dot_r],
                     fill=white, outline=None)
        # Draw label at pre-computed position
        tx = city["label_x"]
        ty = city["label_y"]
        draw.text((tx + shadow, ty + shadow), name, font=font_city, fill=(0, 0, 0))
        draw.text((tx, ty), name, font=font_city, fill=white)


def pick_tick_step(sea_range):
    """Pick a readable axis tick step based on the sea-level range."""
    if sea_range <= 20:
        return 5
    if sea_range <= 60:
        return 10
    if sea_range <= 150:
        return 25
    if sea_range <= 400:
        return 50
    if sea_range <= 1200:
        return 100
    if sea_range <= 3000:
        return 250
    return 500


def build_sea_levels(sea_min, sea_max, sea_step, curve):
    """Build monotonically increasing sea levels using the requested curve."""
    if sea_max <= sea_min:
        return np.array([float(sea_min)], dtype=np.float32)
    frame_count = max(2, int(round((sea_max - sea_min) / sea_step)) + 1)
    t = np.linspace(0.0, 1.0, frame_count, dtype=np.float32)
    if curve == "easein":
        # Gentle acceleration: starts with visible movement, ends 19x faster.
        shaped = 0.1 * t + 0.9 * t ** 2
    elif curve == "quadratic":
        shaped = t ** 2
    elif curve == "cubic":
        shaped = t ** 3
    elif curve == "log":
        # Fast early rise, then flatten (classic logarithmic progression).
        k = 9.0
        shaped = np.log1p(k * t) / np.log1p(k)
    else:
        shaped = t
    return sea_min + (sea_max - sea_min) * shaped


def precompute_scaled_dims(dem_shape):
    """Pre-compute the resize dimensions and offsets for fit_to_frame on GPU."""
    h, w = dem_shape
    scale = min(WIDTH / w, HEIGHT / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    offset_x = (WIDTH - new_w) // 2
    offset_y = (HEIGHT - new_h) // 2
    return new_w, new_h, offset_x, offset_y


@torch.no_grad()
def render_frame_gpu(dem_gpu, is_nodata_gpu, sea_level, terrain_lut_gpu,
                     rivers_mask_gpu, lakes_mask_gpu, lake_rim_weight_gpu,
                     hillshade_gpu, ocean_noise_gpu, shore_t, deep_t, river_t, deep_bg_t):
    """Render a single frame entirely on GPU. Returns HxWx3 float tensor on GPU."""
    # Masks — areas below 0m are protected by dams until sea_level > FLOOD_PROTECTION_M
    is_land = (~is_nodata_gpu) & (dem_gpu >= sea_level)
    is_flooded = (~is_nodata_gpu) & (dem_gpu < sea_level)
    # Output image
    img = torch.zeros_like(hillshade_gpu).unsqueeze(2).expand(-1, -1, 3).contiguous()
    img = img * 0  # zero it out, keep shape

    # Terrain coloring via LUT
    land_elev = torch.clamp(dem_gpu - sea_level, 0, 3000).to(torch.int64)
    land_elev[~is_land] = 0
    img[is_land] = terrain_lut_gpu[land_elev[is_land]].float()
    if LAND_BRIGHTNESS != 1.0 or LAND_SATURATION != 1.0:
        land_rgb = img[is_land]
        if LAND_SATURATION != 1.0:
            luma = (land_rgb[:, 0] * 0.2126 +
                    land_rgb[:, 1] * 0.7152 +
                    land_rgb[:, 2] * 0.0722).unsqueeze(1)
            land_rgb = luma + (land_rgb - luma) * LAND_SATURATION
        if LAND_BRIGHTNESS != 1.0:
            land_rgb = land_rgb * LAND_BRIGHTNESS
        img[is_land] = land_rgb

    # Lake masks
    is_lake = None
    is_lake_unflooded = None
    if lakes_mask_gpu is not None:
        is_lake = lakes_mask_gpu & ~is_nodata_gpu
        is_lake_unflooded = is_lake & ~is_flooded

    # Flooded areas — exclude lake pixels (they keep their own rendering)
    is_flooded_nolake = is_flooded
    if is_lake is not None:
        is_flooded_nolake = is_flooded & ~is_lake
    if torch.any(is_flooded_nolake):
        max_vis_depth = max(50.0, sea_level * 0.5)
        water_depth = sea_level - dem_gpu[is_flooded_nolake]
        depth_norm = torch.clamp(water_depth / max_vis_depth, 0.0, 1.0).unsqueeze(1)
        img[is_flooded_nolake] = shore_t * (1 - depth_norm) + deep_t * depth_norm

    # Rivers on land only (disappear when flooded)
    if rivers_mask_gpu is not None:
        is_river_on_land = rivers_mask_gpu & is_land
        if lakes_mask_gpu is not None:
            is_river_on_land = is_river_on_land & ~lakes_mask_gpu
        if torch.any(is_river_on_land):
            img[is_river_on_land] = river_t

    # All lake pixels: shore-to-deep gradient
    # When flooded, the rim fades out proportionally to how deep the sea is above the lake
    if is_lake is not None and torch.any(is_lake):
        if lake_rim_weight_gpu is not None:
            rim_w = lake_rim_weight_gpu[is_lake].unsqueeze(1)
            # Fade rim toward deep when sea covers the lake
            is_lake_flooded = is_lake & is_flooded
            if torch.any(is_lake_flooded):
                lake_flood_depth = torch.zeros(is_lake.sum(), device=DEVICE)
                lake_flooded_in_lake = is_lake_flooded[is_lake]
                lake_flood_depth[lake_flooded_in_lake] = sea_level - dem_gpu[is_lake][lake_flooded_in_lake]
                fade = torch.clamp(lake_flood_depth / 30.0, 0.0, 1.0).unsqueeze(1)
                rim_w = rim_w * (1.0 - fade)
            img[is_lake] = shore_t * rim_w + deep_t * (1.0 - rim_w)
        else:
            img[is_lake] = deep_t

    # Background ocean
    img[is_nodata_gpu] = deep_bg_t

    # Hillshade on land only (exclude lakes and rivers — no shadow on water)
    hs = hillshade_gpu.unsqueeze(2)
    land_for_hs = is_land
    if is_lake is not None:
        land_for_hs = land_for_hs & ~is_lake
    if rivers_mask_gpu is not None:
        land_for_hs = land_for_hs & ~rivers_mask_gpu
    img = torch.where(land_for_hs.unsqueeze(2), img * hs, img)

    # Ocean noise
    ocean_mask = is_nodata_gpu | is_flooded
    ocean_3d = ocean_mask.unsqueeze(2)
    noise_3d = ocean_noise_gpu.unsqueeze(2)
    img = torch.where(ocean_3d, img * noise_3d, img)

    return torch.clamp(img, 0, 255)


def parse_args():
    parser = argparse.ArgumentParser(description="Sea level rise visualization renderer.")
    parser.add_argument("country", nargs="?",
                        help="Country name (must match Natural Earth, e.g. Germany).")
    parser.add_argument("--country", dest="country_opt", default=None,
                        help="Country name (overrides positional).")
    parser.add_argument("--dem", default=None, help="Path to clipped DEM .tif (overrides --country).")
    parser.add_argument("--full-res", action="store_true",
                        help="Use the full-resolution DEM instead of a resized one.")
    parser.add_argument("--output", default=None, help="Output video path (.mp4).")
    parser.add_argument("--preview", default=None, help="Write a single PNG preview and exit.")
    parser.add_argument("--preview-level", type=float, default=None,
                        help="Sea level for preview in meters (default: SEA_LEVEL_MIN).")
    parser.add_argument("--sea-min", type=float, default=None,
                        help=f"Minimum sea level in meters (default: DEM min elevation).")
    parser.add_argument("--sea-max", type=float, default=None,
                        help=f"Maximum sea level in meters (default: {SEA_LEVEL_MAX}).")
    parser.add_argument("--sea-max-montblanc", action="store_true",
                        help=f"Use Mont Blanc elevation as max sea level ({MONT_BLANC_ELEVATION_M}m).")
    parser.add_argument("--sea-step", type=float, default=SEA_LEVEL_STEP,
                        help=f"Step in meters used to determine frame count (default: {SEA_LEVEL_STEP}).")
    parser.add_argument("--sea-curve", choices=["linear", "easein", "quadratic", "cubic", "log"], default="easein",
                        help="Sea-level growth curve over time (default: easein).")
    parser.add_argument("--ui-tick-step", type=int, default=None,
                        help="Fixed tick step (meters) for the bottom scale. Default: auto.")
    parser.add_argument("--width", type=int, default=None,
                        help="Video width in pixels (default: 3840).")
    parser.add_argument("--height", type=int, default=None,
                        help="Video height in pixels (default: 2160).")
    parser.add_argument("--fps", type=int, default=None,
                        help="Frames per second (default: 60).")
    parser.add_argument("--duration", type=float, default=None,
                        help="Video duration in seconds. Overrides --sea-step.")
    parser.add_argument("--cities", type=int, default=0,
                        help="Show top N cities by population (e.g. --cities 10). Default: 0 (off).")
    return parser.parse_args()


def find_resized_dem(country_dir, country_slug, target_w, target_h):
    pattern = re.compile(rf"^{re.escape(country_slug)}_(\d+)x(\d+)_dem_clipped\.tif$")
    candidates = []
    for base_dir in [country_dir, DATA_DIR]:
        if not os.path.isdir(base_dir):
            continue
        for name in os.listdir(base_dir):
            m = pattern.match(name)
            if not m:
                continue
            w, h = int(m.group(1)), int(m.group(2))
            if (w, h) == (1920, 1080):
                # Ignore legacy stretched 1080p outputs
                continue
            candidates.append((w, h, os.path.join(base_dir, name)))
    if not candidates:
        return None
    within = [c for c in candidates if c[0] <= target_w and c[1] <= target_h]
    if within:
        within.sort(key=lambda c: (c[0] * c[1], c[0], c[1]), reverse=True)
        return within[0][2]
    candidates.sort(key=lambda c: (c[0] * c[1], c[0], c[1]))
    return candidates[0][2]


def main():
    args = parse_args()
    if args.country_opt and args.country and args.country_opt != args.country:
        print("Warning: both positional and --country provided; using --country.")
    # Override globals if CLI args provided
    global WIDTH, HEIGHT, FPS, UI_SCALE
    if args.width is not None:
        WIDTH = args.width
    if args.height is not None:
        HEIGHT = args.height
    if args.fps is not None:
        FPS = args.fps
    UI_SCALE = min(WIDTH / 1920, HEIGHT / 1080)

    country_name = args.country_opt or args.country or DEFAULT_COUNTRY
    country_slug = slugify(country_name)
    country_dir = os.path.join(DATA_DIR, country_slug)
    default_dem = os.path.join(country_dir, f"{country_slug}_dem_clipped.tif")
    dem_path = args.dem or default_dem
    if not args.dem and not args.full_res:
        max_w = int(WIDTH * RESIZED_MAX_SCALE)
        max_h = int(HEIGHT * RESIZED_MAX_SCALE)
        resized_dem = find_resized_dem(country_dir, country_slug, max_w, max_h)
        if resized_dem:
            print(f"Note: using resized DEM: {resized_dem}")
            dem_path = resized_dem
    if not args.dem and not os.path.exists(dem_path):
        legacy_dem = os.path.join(DATA_DIR, f"{country_slug}_dem_clipped.tif")
        if os.path.exists(legacy_dem):
            print(f"Note: using legacy DEM path: {legacy_dem}")
            dem_path = legacy_dem
    dem_tag = country_slug
    if dem_path != default_dem:
        dem_base = os.path.splitext(os.path.basename(dem_path))[0]
        dem_tag = slugify(dem_base.replace("_dem_clipped", ""))
    is_resized_dem = re.search(r"_\d+x\d+_dem_clipped\.tif$", os.path.basename(dem_path)) is not None
    is_etopo_dem = "etopo" in os.path.basename(dem_path).lower()
    if is_etopo_dem:
        # Gentler land boost for coarse ETOPO DEMs (avoid oversaturation)
        globals()["LAND_BRIGHTNESS"] = LAND_BRIGHTNESS_ETOPO
        globals()["LAND_SATURATION"] = LAND_SATURATION_ETOPO
    if args.output:
        output_video = args.output
    else:
        name_tag = dem_tag if args.dem else country_slug
        output_video = os.path.join(SCRIPT_DIR, f"sea_level_rise_{name_tag}.mp4")
    preview_path = args.preview
    preview_level = args.preview_level
    data_dir = os.path.dirname(dem_path)
    print(f"=== Sea Level Rise Visualization - {country_name} (GPU) ===\n")
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {mem_gb:.1f} GB")
    print(f"DEM: {dem_path}")
    print(f"Output: {output_video}")
    if preview_path:
        print(f"Preview: {preview_path}")
    print()

    # Load data
    print("Loading DEM data...")
    dem_np, bounds, dem_elev_min, dem_elev_max = load_dem_data(dem_path)

    # Compute pixel size for resolution-aware hillshade
    with rasterio.open(dem_path) as src:
        pixel_m = abs(src.transform.a) * 111000
    print(f"Pixel size: ~{pixel_m:.0f}m")

    print("Loading water bodies...")
    water_data = load_water_bodies(dem_path, dem_tag, data_dir)
    if isinstance(water_data, tuple):
        rivers_np, lakes_np = water_data
    else:
        rivers_np = water_data
        lakes_np = None

    print("Creating terrain colormap...")
    terrain_lut_gpu = create_terrain_lut()

    # Move data to GPU
    print("Uploading data to GPU...")
    is_nodata_np = np.isnan(dem_np)
    dem_clean = np.where(is_nodata_np, 0, dem_np)

    dem_gpu = torch.from_numpy(dem_clean).to(DEVICE, dtype=torch.float32)
    is_nodata_gpu = torch.from_numpy(is_nodata_np).to(DEVICE, dtype=torch.bool)
    rivers_mask_gpu = torch.from_numpy(rivers_np > 0).to(DEVICE, dtype=torch.bool)
    lakes_mask_gpu = None
    lake_rim_weight_gpu = None
    if lakes_np is not None:
        lakes_mask_gpu = torch.from_numpy(lakes_np > 0).to(DEVICE, dtype=torch.bool)
        scaled_rim_px = max(0, round(LAKE_SHORE_PX * (90.0 / pixel_m)))
        if scaled_rim_px > 0:
            lake_rim_weight_gpu = build_lake_rim_weight(lakes_mask_gpu, scaled_rim_px)

    # Pre-allocate color tensors on GPU (avoid re-creating each frame)
    shore_t = torch.tensor(OCEAN_COLOR_SHORE, dtype=torch.float32, device=DEVICE)
    deep_t = torch.tensor(OCEAN_COLOR_DEEP, dtype=torch.float32, device=DEVICE)
    river_t = torch.tensor(RIVER_COLOR, dtype=torch.float32, device=DEVICE)
    deep_bg_t = torch.tensor(OCEAN_COLOR_DEEP, dtype=torch.float32, device=DEVICE)

    print("Computing hillshade on GPU...")
    hillshade_gpu = compute_hillshade_gpu(dem_np, pixel_size_m=pixel_m)
    if is_resized_dem:
        if HILLSHADE_STRENGTH_RESIZED != 1.0:
            hillshade_gpu = 1.0 - HILLSHADE_STRENGTH_RESIZED + HILLSHADE_STRENGTH_RESIZED * hillshade_gpu
        if HILLSHADE_BLUR_ITERS_RESIZED > 0:
            hillshade_gpu = blur_hillshade_gpu(hillshade_gpu, HILLSHADE_BLUR_ITERS_RESIZED)
        if HILLSHADE_GAIN_RESIZED != 1.0:
            hillshade_gpu = torch.clamp(hillshade_gpu * HILLSHADE_GAIN_RESIZED, 0.0, 1.0)

    if is_etopo_dem:
        if HILLSHADE_STRENGTH_ETOPO != 1.0:
            hillshade_gpu = 1.0 - HILLSHADE_STRENGTH_ETOPO + HILLSHADE_STRENGTH_ETOPO * hillshade_gpu
        if HILLSHADE_GAIN_ETOPO != 1.0:
            hillshade_gpu = torch.clamp(hillshade_gpu * HILLSHADE_GAIN_ETOPO, 0.0, 1.0)

    print("Generating ocean noise...")
    from scipy.ndimage import gaussian_filter
    noise_np = np.random.RandomState(42).uniform(0.95, 1.05, dem_np.shape).astype(np.float32)
    noise_np = gaussian_filter(noise_np, sigma=5)
    ocean_noise_gpu = torch.from_numpy(noise_np).to(DEVICE, dtype=torch.float32)

    # Sea levels — auto-detect from DEM if not explicitly set
    sea_min = float(args.sea_min) if args.sea_min is not None else dem_elev_min
    if args.sea_max_montblanc:
        sea_max = float(MONT_BLANC_ELEVATION_M)
    elif args.sea_max is not None:
        sea_max = float(args.sea_max)
    else:
        # Use true peak elevation for Austria (ETOPO is too coarse for Großglockner)
        if country_name.lower() == "austria" and dem_elev_max < GROSSGLOCKNER_ELEVATION_M:
            sea_max = float(GROSSGLOCKNER_ELEVATION_M)
        else:
            sea_max = dem_elev_max
    sea_step = float(args.sea_step)
    if args.duration is not None:
        total_frames_target = max(2, int(round(args.duration * FPS)))
        sea_step = max(0.001, (sea_max - sea_min) / (total_frames_target - 1))
        print(f"Duration {args.duration}s -> {total_frames_target} frames, sea_step={sea_step:.3f}m")
    if sea_step <= 0:
        print("ERROR: --sea-step must be > 0.")
        sys.exit(1)
    if sea_max < sea_min:
        print("ERROR: --sea-max must be >= --sea-min.")
        sys.exit(1)
    sea_levels = build_sea_levels(sea_min, sea_max, sea_step, args.sea_curve)
    rise_frames = len(sea_levels)
    fade_in_frames = int(FADE_DURATION_SEC * FPS)
    hold_start_frames = int(HOLD_START_SEC * FPS)
    hold_end_frames = int(HOLD_END_SEC * FPS)
    fade_out_frames = int(FADE_DURATION_SEC * FPS)
    total_frames = fade_in_frames + hold_start_frames + rise_frames + hold_end_frames + fade_out_frames
    duration_sec = total_frames / FPS
    print(
        f"\nSea levels: {sea_min:.1f}m -> {sea_max:.1f}m "
        f"({args.sea_curve}, base step {sea_step}m)"
    )
    print(f"Rise frames: {rise_frames}, total frames: {total_frames}")
    print(f"  Fade-in: {FADE_DURATION_SEC}s, Hold start: {HOLD_START_SEC}s, "
          f"Hold end: {HOLD_END_SEC}s, Fade-out: {FADE_DURATION_SEC}s")
    print(f"Video duration: {duration_sec:.1f}s at {FPS}fps")

    # Load fonts
    fonts = load_fonts()
    font_city = None
    if args.cities > 0:
        scale = UI_SCALE
        for fp in ["C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf",
                   "C:/Windows/Fonts/calibri.ttf"]:
            if os.path.exists(fp):
                try:
                    font_city = ImageFont.truetype(fp, int(28 * scale))
                    break
                except Exception:
                    continue
        if font_city is None:
            font_city = ImageFont.load_default()

    # Pre-compute resize dimensions
    new_w, new_h, off_x, off_y = precompute_scaled_dims(dem_np.shape)

    # Load cities if requested
    city_data = []
    if args.cities > 0:
        print(f"Loading top {args.cities} cities...")
        city_data = load_cities(country_name, dem_path, args.cities)
        city_data = project_cities_to_frame(city_data, dem_path, new_w, new_h, off_x, off_y)
        resolve_city_label_positions(city_data, font_city)

    # Pre-render background frame
    bg_frame = Image.new('RGB', (WIDTH, HEIGHT), tuple(OCEAN_COLOR_DEEP))
    bg_array = np.array(bg_frame)

    def resize_to_frame(img_gpu):
        img_chw = img_gpu.permute(2, 0, 1).unsqueeze(0)
        src_h, src_w = img_chw.shape[2], img_chw.shape[3]
        downsampling = new_h < src_h or new_w < src_w
        if SUPERSAMPLE > 1.0:
            ss_w = max(1, int(new_w * SUPERSAMPLE))
            ss_h = max(1, int(new_h * SUPERSAMPLE))
            img_resized = F.interpolate(
                img_chw, size=(ss_h, ss_w),
                mode='bicubic', align_corners=False
            )
            if downsampling:
                img_resized = F.interpolate(
                    img_resized, size=(new_h, new_w),
                    mode='area'
                )
            else:
                img_resized = F.interpolate(
                    img_resized, size=(new_h, new_w),
                    mode='bicubic', align_corners=False, antialias=True
                )
        else:
            if downsampling:
                img_resized = F.interpolate(
                    img_chw, size=(new_h, new_w),
                    mode='area'
                )
            else:
                img_resized = F.interpolate(
                    img_chw, size=(new_h, new_w),
                    mode='bicubic', align_corners=False, antialias=True
                )
        img_resized = img_resized.squeeze(0).permute(1, 2, 0)
        img_resized = torch.clamp(img_resized, 0, 255)
        return img_resized.to(torch.uint8).cpu().numpy()

    if preview_path:
        sl = float(sea_min if preview_level is None else preview_level)

        img_gpu = render_frame_gpu(
            dem_gpu, is_nodata_gpu, sl, terrain_lut_gpu,
            rivers_mask_gpu, lakes_mask_gpu, lake_rim_weight_gpu,
            hillshade_gpu, ocean_noise_gpu, shore_t, deep_t, river_t, deep_bg_t
        )

        img_small = resize_to_frame(img_gpu)

        frame_array = bg_array.copy()
        frame_array[off_y:off_y + new_h, off_x:off_x + new_w] = img_small

        frame_img = Image.fromarray(frame_array)
        draw_text_on_frame(frame_img, sl, fonts, sea_min, sea_max)
        if city_data and font_city:
            draw = ImageDraw.Draw(frame_img)
            draw_cities_on_frame(draw, city_data, sl, font_city)
        frame_img.save(preview_path)
        print(f"Preview saved to: {preview_path}")
        return

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
    ffmpeg_log = os.path.join(os.path.dirname(output_video) or ".", "ffmpeg_log.txt")
    ffmpeg_log_fh = open(ffmpeg_log, "w")
    ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE,
                                    stdout=subprocess.DEVNULL, stderr=ffmpeg_log_fh)

    print("Generating frames...\n")
    t_start = time.time()

    # Phase boundaries (cumulative frame indices)
    phase1_end = fade_in_frames                          # end of fade-in
    phase2_end = phase1_end + hold_start_frames           # end of hold-start
    phase3_end = phase2_end + rise_frames                 # end of sea-level rise
    phase4_end = phase3_end + hold_end_frames             # end of hold-end
    # phase5 = fade-out until total_frames

    last_img_gpu = None  # cache for hold phases

    for i in range(total_frames):
        # Determine sea level for this frame
        if i < phase2_end:
            sl = sea_min          # fade-in + hold-start: stay at sea_min
            rise_idx = 0
        elif i < phase3_end:
            rise_idx = i - phase2_end
            sl = float(sea_levels[rise_idx])
        else:
            sl = sea_max          # hold-end + fade-out: stay at sea_max
            rise_idx = rise_frames - 1

        # Only re-render when sea level actually changes
        need_render = (last_img_gpu is None
                       or (i >= phase2_end and i < phase3_end)  # during rise
                       or i == phase2_end  # first rise frame
                       or i == phase3_end) # first hold-end frame
        if need_render:
            img_gpu = render_frame_gpu(
                dem_gpu, is_nodata_gpu, sl, terrain_lut_gpu,
                rivers_mask_gpu, lakes_mask_gpu, lake_rim_weight_gpu,
                hillshade_gpu, ocean_noise_gpu, shore_t, deep_t, river_t, deep_bg_t
            )
            img_small = resize_to_frame(img_gpu)
            cached_frame = bg_array.copy()
            cached_frame[off_y:off_y + new_h, off_x:off_x + new_w] = img_small
            last_img_gpu = cached_frame

        # Compose frame
        frame_array = last_img_gpu.copy()
        frame_img = Image.fromarray(frame_array)
        draw_text_on_frame(frame_img, sl, fonts, sea_min, sea_max)
        if city_data and font_city:
            draw = ImageDraw.Draw(frame_img)
            draw_cities_on_frame(draw, city_data, sl, font_city)

        # Fade-to-black
        if i < fade_in_frames:
            alpha = i / fade_in_frames
            faded = (np.array(frame_img, dtype=np.float32) * alpha).astype(np.uint8)
            frame_img = Image.fromarray(faded)
        elif i >= phase4_end:
            frames_into_fade = i - phase4_end
            alpha = 1.0 - frames_into_fade / fade_out_frames
            alpha = max(0.0, alpha)
            faded = (np.array(frame_img, dtype=np.float32) * alpha).astype(np.uint8)
            frame_img = Image.fromarray(faded)

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
    ffmpeg_log_fh.close()

    # Read ffmpeg log
    ffmpeg_stderr = ""
    if os.path.exists(ffmpeg_log):
        with open(ffmpeg_log, "r", errors="replace") as f:
            ffmpeg_stderr = f.read()

    elapsed_total = time.time() - t_start
    print(f"\nAll frames generated in {elapsed_total:.1f}s ({total_frames / elapsed_total:.1f} fps)")

    if ffmpeg_stderr:
        # Show last 20 lines of ffmpeg output for debugging
        lines = ffmpeg_stderr.strip().splitlines()
        if len(lines) > 20:
            print(f"\n[ffmpeg stderr - last 20 of {len(lines)} lines]")
            for l in lines[-20:]:
                print(f"  {l}")
        else:
            print(f"\n[ffmpeg stderr]")
            for l in lines:
                print(f"  {l}")

    if ffmpeg_proc.returncode == 0:
        size_mb = os.path.getsize(output_video) / (1024 * 1024)
        print(f"\nVideo saved to: {output_video}")
        print(f"File size: {size_mb:.1f} MB")
    else:
        print(f"FFmpeg failed with return code {ffmpeg_proc.returncode}")
        if ffmpeg_stderr:
            for l in ffmpeg_stderr.strip().splitlines()[-10:]:
                print(f"  {l}")
        sys.exit(1)

    print("\n=== Done! ===")


if __name__ == "__main__":
    main()
