"""
Build a complete world scene in Blender from a heightmap.

Usage example:
  blender --background --factory-startup --python blender_setup_world.py -- \
    --heightmap data/world/world_height_32k.tif \
    --out data/world/world_setup_32k.blend
"""

import argparse
import math
import os
import sys

import bpy


# Terrain colors from generate_video.py
TERRAIN_COLORS = [
    (0, (30, 120, 50)),
    (50, (60, 160, 60)),
    (100, (100, 180, 70)),
    (200, (160, 200, 80)),
    (300, (200, 210, 100)),
    (400, (220, 200, 80)),
    (500, (220, 170, 60)),
    (600, (210, 140, 50)),
    (800, (190, 100, 40)),
    (1000, (170, 70, 30)),
    (1500, (140, 50, 25)),
    (2000, (120, 100, 90)),
    (3000, (200, 200, 210)),
]


def rgb01(rgb):
    return (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0, 1.0)


def set_input_if_exists(node, names, value):
    for name in names:
        sock = node.inputs.get(name)
        if sock is not None:
            sock.default_value = value
            return True
    return False


def set_displacement_method(material):
    # Blender version-safe
    if hasattr(material, "displacement_method"):
        try:
            material.displacement_method = "BOTH"
            return
        except Exception:
            pass
    if hasattr(material, "cycles") and hasattr(material.cycles, "displacement_method"):
        try:
            material.cycles.displacement_method = "BOTH"
        except Exception:
            pass


def enable_cycles_gpu(scene):
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"

    backend_selected = "CPU-fallback"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for backend in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
        try:
            prefs.compute_device_type = backend
            if hasattr(prefs, "get_devices"):
                prefs.get_devices()
            found_gpu = False
            for dev in prefs.devices:
                if dev.type != "CPU":
                    dev.use = True
                    found_gpu = True
                else:
                    dev.use = False
            if found_gpu:
                backend_selected = backend
                break
        except Exception:
            continue

    print(f"[GPU] Cycles backend: {backend_selected}")
    # Use "Standard" for map-like colors (Filmic can look washed out for this style).
    try:
        scene.view_settings.view_transform = "Standard"
    except Exception:
        pass
    try:
        scene.view_settings.look = "None"
    except Exception:
        pass
    try:
        scene.view_settings.exposure = 0.0
        scene.view_settings.gamma = 1.0
    except Exception:
        pass
    # Reduce viewport noise where supported.
    if hasattr(scene.cycles, "use_preview_denoising"):
        scene.cycles.use_preview_denoising = True
    if hasattr(scene.cycles, "use_denoising"):
        scene.cycles.use_denoising = True


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    # Remove orphan materials/images from default startup
    for mat in list(bpy.data.materials):
        if mat.users == 0:
            bpy.data.materials.remove(mat)
    for img in list(bpy.data.images):
        if img.users == 0:
            bpy.data.images.remove(img)


def build_terrain_ramp(color_ramp):
    while len(color_ramp.elements) > 2:
        color_ramp.elements.remove(color_ramp.elements[-1])

    color_ramp.elements[0].position = 0.0
    color_ramp.elements[0].color = rgb01(TERRAIN_COLORS[0][1])
    color_ramp.elements[1].position = 1.0
    color_ramp.elements[1].color = rgb01(TERRAIN_COLORS[-1][1])

    max_elev = 3000.0
    for elev, col in TERRAIN_COLORS[1:-1]:
        e = color_ramp.elements.new(max(0.0, min(1.0, elev / max_elev)))
        e.color = rgb01(col)


def create_land_material(height_img, cfg):
    mat = bpy.data.materials.new("Earth_Land_Mat")
    mat.use_nodes = True
    set_displacement_method(mat)

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    n_out = nodes.new("ShaderNodeOutputMaterial")
    n_bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    n_texcoord = nodes.new("ShaderNodeTexCoord")
    n_tex = nodes.new("ShaderNodeTexImage")
    n_cmp_nodata = nodes.new("ShaderNodeMath")
    n_one_minus = nodes.new("ShaderNodeMath")
    n_mul_keep = nodes.new("ShaderNodeMath")
    n_mul_fill = nodes.new("ShaderNodeMath")
    n_add_clean = nodes.new("ShaderNodeMath")
    n_map_terrain = nodes.new("ShaderNodeMapRange")
    n_map_disp = nodes.new("ShaderNodeMapRange")
    n_ramp_terrain = nodes.new("ShaderNodeValToRGB")
    n_hsv = nodes.new("ShaderNodeHueSaturation")
    n_bc = nodes.new("ShaderNodeBrightContrast")
    n_disp = nodes.new("ShaderNodeDisplacement")

    n_texcoord.location = (-1400, 260)
    n_tex.location = (-1200, 220)
    n_cmp_nodata.location = (-1000, -120)
    n_one_minus.location = (-820, -120)
    n_mul_keep.location = (-640, 40)
    n_mul_fill.location = (-640, -120)
    n_add_clean.location = (-460, -40)
    n_map_terrain.location = (-260, 260)
    n_map_disp.location = (-260, -120)
    n_ramp_terrain.location = (-720, 260)
    n_hsv.location = (-120, 260)
    n_bc.location = (90, 260)
    n_bsdf.location = (380, 180)
    n_disp.location = (380, -120)
    n_out.location = (640, 80)

    # Height texture
    n_tex.image = height_img
    n_tex.interpolation = "Cubic"
    # Avoid UV seam artifacts on equirectangular world maps.
    n_tex.extension = "REPEAT"
    n_tex.projection = "FLAT"
    n_tex.image.colorspace_settings.name = "Non-Color"

    # Replace nodata samples by a safe fill value before shading/displacement.
    n_cmp_nodata.operation = "COMPARE"
    n_cmp_nodata.inputs[1].default_value = cfg["nodata_value"]
    n_cmp_nodata.inputs[2].default_value = cfg["nodata_epsilon"]

    n_one_minus.operation = "SUBTRACT"
    n_one_minus.inputs[0].default_value = 1.0

    n_mul_keep.operation = "MULTIPLY"
    n_mul_fill.operation = "MULTIPLY"
    n_mul_fill.inputs[0].default_value = cfg["nodata_fill_m"]
    n_add_clean.operation = "ADD"

    # Terrain color mapping
    n_map_terrain.clamp = True
    n_map_terrain.inputs["From Min"].default_value = 0.0
    n_map_terrain.inputs["From Max"].default_value = 3000.0
    n_map_terrain.inputs["To Min"].default_value = 0.0
    n_map_terrain.inputs["To Max"].default_value = 1.0
    build_terrain_ramp(n_ramp_terrain.color_ramp)

    # Clamp displacement range to stable DEM values.
    n_map_disp.clamp = True
    n_map_disp.inputs["From Min"].default_value = cfg["disp_min_m"]
    n_map_disp.inputs["From Max"].default_value = cfg["disp_max_m"]
    n_map_disp.inputs["To Min"].default_value = cfg["disp_min_m"]
    n_map_disp.inputs["To Max"].default_value = cfg["disp_max_m"]

    n_hsv.inputs["Saturation"].default_value = 1.25
    n_hsv.inputs["Value"].default_value = 1.03
    n_bc.inputs["Bright"].default_value = 0.02
    n_bc.inputs["Contrast"].default_value = 0.08

    set_input_if_exists(n_bsdf, ["Roughness"], 0.55)
    set_input_if_exists(n_bsdf, ["Specular IOR Level", "Specular"], 0.4)

    meters_to_units = cfg["vertical_exaggeration"] / cfg["earth_radius_m"]
    n_disp.inputs["Scale"].default_value = meters_to_units
    n_disp.inputs["Midlevel"].default_value = 0.0

    links.new(n_texcoord.outputs["UV"], n_tex.inputs["Vector"])

    links.new(n_tex.outputs["Color"], n_cmp_nodata.inputs[0])
    links.new(n_cmp_nodata.outputs["Value"], n_one_minus.inputs[1])
    links.new(n_tex.outputs["Color"], n_mul_keep.inputs[0])
    links.new(n_one_minus.outputs["Value"], n_mul_keep.inputs[1])
    links.new(n_cmp_nodata.outputs["Value"], n_mul_fill.inputs[1])
    links.new(n_mul_keep.outputs["Value"], n_add_clean.inputs[0])
    links.new(n_mul_fill.outputs["Value"], n_add_clean.inputs[1])

    links.new(n_add_clean.outputs["Value"], n_map_terrain.inputs["Value"])
    links.new(n_add_clean.outputs["Value"], n_map_disp.inputs["Value"])
    links.new(n_map_terrain.outputs["Result"], n_ramp_terrain.inputs["Fac"])
    links.new(n_ramp_terrain.outputs["Color"], n_hsv.inputs["Color"])
    links.new(n_hsv.outputs["Color"], n_bc.inputs["Color"])

    links.new(n_bc.outputs["Color"], n_bsdf.inputs["Base Color"])
    links.new(n_bsdf.outputs["BSDF"], n_out.inputs["Surface"])

    links.new(n_map_disp.outputs["Result"], n_disp.inputs["Height"])
    links.new(n_disp.outputs["Displacement"], n_out.inputs["Displacement"])

    return mat


def create_water_material(cfg):
    mat = bpy.data.materials.new("Earth_Water_Mat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    n_out = nodes.new("ShaderNodeOutputMaterial")
    n_bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    n_noise = nodes.new("ShaderNodeTexNoise")
    n_bump = nodes.new("ShaderNodeBump")
    n_layer = nodes.new("ShaderNodeLayerWeight")
    n_mix_col = nodes.new("ShaderNodeMixRGB")
    n_vol_abs = nodes.new("ShaderNodeVolumeAbsorption")

    n_layer.location = (-480, 180)
    n_mix_col.location = (-260, 180)
    n_noise.location = (-480, -80)
    n_bump.location = (-260, -80)
    n_vol_abs.location = (-20, -140)
    n_bsdf.location = (-20, 100)
    n_out.location = (220, 100)

    n_mix_col.blend_type = "MIX"
    n_mix_col.inputs[0].default_value = 1.0
    n_mix_col.inputs[1].default_value = tuple(cfg["water_color_shallow"])
    n_mix_col.inputs[2].default_value = tuple(cfg["water_color_deep"])

    # Physical-ish water setup with depth fade through volume absorption.
    set_input_if_exists(n_bsdf, ["Transmission", "Transmission Weight"], cfg["water_transmission"])
    set_input_if_exists(n_bsdf, ["Roughness"], cfg["water_roughness"])
    set_input_if_exists(n_bsdf, ["IOR"], cfg["water_ior"])
    set_input_if_exists(n_bsdf, ["Specular IOR Level", "Specular"], cfg["water_specular"])

    n_noise.inputs["Scale"].default_value = cfg["water_noise_scale"]
    n_noise.inputs["Detail"].default_value = 1.0
    n_noise.inputs["Roughness"].default_value = 0.35
    n_bump.inputs["Strength"].default_value = cfg["water_bump_strength"]
    n_bump.inputs["Distance"].default_value = 1.0
    n_vol_abs.inputs["Color"].default_value = tuple(cfg["water_absorption_color"])
    n_vol_abs.inputs["Density"].default_value = cfg["water_absorption_density"]

    links.new(n_layer.outputs["Facing"], n_mix_col.inputs["Fac"])
    links.new(n_mix_col.outputs["Color"], n_bsdf.inputs["Base Color"])
    links.new(n_noise.outputs["Fac"], n_bump.inputs["Height"])
    links.new(n_bump.outputs["Normal"], n_bsdf.inputs["Normal"])
    links.new(n_bsdf.outputs["BSDF"], n_out.inputs["Surface"])
    links.new(n_vol_abs.outputs["Volume"], n_out.inputs["Volume"])

    return mat


def atmosphere_radius(cfg):
    # Atmosphere thickness should follow planet scale, not terrain exaggeration.
    meters_to_units = cfg["atmo_height_exag"] / cfg["earth_radius_m"]
    return cfg["land_radius"] + cfg["atmo_height_m"] * meters_to_units


def create_atmosphere_material(cfg):
    mat = bpy.data.materials.new("Earth_Atmosphere_Mat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    n_out = nodes.new("ShaderNodeOutputMaterial")
    n_layer = nodes.new("ShaderNodeLayerWeight")
    n_map = nodes.new("ShaderNodeMapRange")
    n_emission = nodes.new("ShaderNodeEmission")
    n_transparent = nodes.new("ShaderNodeBsdfTransparent")
    n_mix = nodes.new("ShaderNodeMixShader")

    n_layer.location = (-620, 100)
    n_map.location = (-420, 100)
    n_emission.location = (-220, 180)
    n_transparent.location = (-220, 20)
    n_mix.location = (0, 100)
    n_out.location = (220, 100)

    n_map.clamp = True
    n_map.inputs["From Min"].default_value = 0.55
    n_map.inputs["From Max"].default_value = 1.0
    n_map.inputs["To Min"].default_value = 0.0
    n_map.inputs["To Max"].default_value = 1.0

    n_emission.inputs["Color"].default_value = tuple(cfg["atmo_scatter_color"])
    n_emission.inputs["Strength"].default_value = max(0.2, cfg["atmo_scatter_density"] * 120.0)

    links.new(n_layer.outputs["Facing"], n_map.inputs["Value"])
    links.new(n_map.outputs["Result"], n_mix.inputs["Fac"])
    links.new(n_transparent.outputs["BSDF"], n_mix.inputs[1])
    links.new(n_emission.outputs["Emission"], n_mix.inputs[2])
    links.new(n_mix.outputs["Shader"], n_out.inputs["Surface"])

    return mat


def water_radius_for_sea(cfg, sea_level_m):
    meters_to_units = cfg["vertical_exaggeration"] / cfg["earth_radius_m"]
    offset_units = cfg["water_surface_offset_m"] * meters_to_units
    return cfg["land_radius"] + offset_units + sea_level_m * meters_to_units


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heightmap", required=True, help="Path to world heightmap tif.")
    parser.add_argument("--out", required=True, help="Output .blend path.")
    parser.add_argument("--sea-level", type=float, default=0.0, help="Sea level in meters.")
    parser.add_argument("--sea-end", type=float, default=60.0, help="End sea level in meters for animation.")
    parser.add_argument("--animate", action="store_true", help="Create sea-level animation.")
    parser.add_argument("--frame-start", type=int, default=1)
    parser.add_argument("--frame-end", type=int, default=240)
    parser.add_argument("--vertical-exag", type=float, default=16.0, help="Vertical exaggeration factor.")
    parser.add_argument("--segments", type=int, default=512)
    parser.add_argument("--rings", type=int, default=256)
    parser.add_argument("--subdiv-view", type=int, default=1)
    parser.add_argument("--subdiv-render", type=int, default=3)
    parser.add_argument("--water-subdiv-view", type=int, default=2)
    parser.add_argument("--water-subdiv-render", type=int, default=4)
    parser.add_argument("--viewport-dicing", type=float, default=4.0)
    parser.add_argument("--render-dicing", type=float, default=1.0)
    parser.add_argument("--viewport-samples", type=int, default=32)
    parser.add_argument("--render-samples", type=int, default=192)
    parser.add_argument("--sea-offset-m", type=float, default=0.5)
    parser.add_argument("--water-transmission", type=float, default=1.0)
    parser.add_argument("--water-roughness", type=float, default=0.06)
    parser.add_argument("--water-ior", type=float, default=1.333)
    parser.add_argument("--water-specular", type=float, default=0.28)
    parser.add_argument("--water-noise-scale", type=float, default=56.0)
    parser.add_argument("--water-bump-strength", type=float, default=0.00008)
    parser.add_argument("--water-absorption-density", type=float, default=550.0)
    parser.add_argument("--water-absorption-r", type=float, default=0.20)
    parser.add_argument("--water-absorption-g", type=float, default=0.42)
    parser.add_argument("--water-absorption-b", type=float, default=0.70)
    parser.add_argument("--water-color-shallow-r", type=float, default=0.13)
    parser.add_argument("--water-color-shallow-g", type=float, default=0.34)
    parser.add_argument("--water-color-shallow-b", type=float, default=0.62)
    parser.add_argument("--water-color-deep-r", type=float, default=0.05)
    parser.add_argument("--water-color-deep-g", type=float, default=0.13)
    parser.add_argument("--water-color-deep-b", type=float, default=0.30)
    parser.add_argument("--atmo-enable", dest="atmo_enable", action="store_true")
    parser.add_argument("--no-atmo", dest="atmo_enable", action="store_false")
    parser.add_argument("--atmo-render-only", dest="atmo_render_only", action="store_true")
    parser.add_argument("--atmo-show-viewport", dest="atmo_render_only", action="store_false")
    parser.add_argument("--atmo-height-km", type=float, default=85.0)
    parser.add_argument("--atmo-height-exag", type=float, default=1.0)
    parser.add_argument("--atmo-scatter-density", type=float, default=0.0015)
    parser.add_argument("--atmo-absorption-density", type=float, default=0.00025)
    parser.add_argument("--atmo-anisotropy", type=float, default=0.6)
    parser.add_argument("--atmo-scatter-r", type=float, default=0.42)
    parser.add_argument("--atmo-scatter-g", type=float, default=0.63)
    parser.add_argument("--atmo-scatter-b", type=float, default=1.0)
    parser.add_argument("--atmo-absorption-r", type=float, default=0.16)
    parser.add_argument("--atmo-absorption-g", type=float, default=0.28)
    parser.add_argument("--atmo-absorption-b", type=float, default=0.48)
    parser.add_argument("--atmo-subdiv-view", type=int, default=1)
    parser.add_argument("--atmo-subdiv-render", type=int, default=2)
    parser.add_argument("--nodata-value", type=float, default=-9999.0)
    parser.add_argument("--nodata-epsilon", type=float, default=2.0)
    parser.add_argument("--nodata-fill", type=float, default=0.0)
    parser.add_argument("--disp-min", type=float, default=-11000.0)
    parser.add_argument("--disp-max", type=float, default=9000.0)
    parser.add_argument("--experimental", dest="experimental", action="store_true")
    parser.add_argument("--no-experimental", dest="experimental", action="store_false")
    parser.add_argument("--adaptive-subdiv", dest="adaptive_subdiv", action="store_true")
    parser.add_argument("--no-adaptive-subdiv", dest="adaptive_subdiv", action="store_false")
    parser.set_defaults(experimental=True, adaptive_subdiv=True, atmo_enable=True, atmo_render_only=True)

    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    return parser.parse_args(argv)


def main():
    args = parse_args()
    root_dir = os.path.dirname(os.path.abspath(__file__))
    heightmap_path = args.heightmap
    out_path = args.out
    if not os.path.isabs(heightmap_path):
        heightmap_path = os.path.join(root_dir, heightmap_path)
    if not os.path.isabs(out_path):
        out_path = os.path.join(root_dir, out_path)

    if not os.path.exists(heightmap_path):
        raise FileNotFoundError(f"Heightmap not found: {heightmap_path}")

    cfg = {
        "sea_level_m": float(args.sea_level),
        "max_vis_depth_m": 120.0,
        "vertical_exaggeration": float(args.vertical_exag),
        "earth_radius_m": 6_371_000.0,
        "land_radius": 1.0,
        # Small physical offset (meters) to avoid z-fighting at exact sea level.
        "water_surface_offset_m": float(args.sea_offset_m),
        "water_transmission": float(args.water_transmission),
        "water_roughness": float(args.water_roughness),
        "water_ior": float(args.water_ior),
        "water_specular": float(args.water_specular),
        "water_noise_scale": float(args.water_noise_scale),
        "water_bump_strength": float(args.water_bump_strength),
        "water_absorption_density": float(args.water_absorption_density),
        "water_absorption_color": (
            float(args.water_absorption_r),
            float(args.water_absorption_g),
            float(args.water_absorption_b),
            1.0,
        ),
        "water_color_shallow": (
            float(args.water_color_shallow_r),
            float(args.water_color_shallow_g),
            float(args.water_color_shallow_b),
            1.0,
        ),
        "water_color_deep": (
            float(args.water_color_deep_r),
            float(args.water_color_deep_g),
            float(args.water_color_deep_b),
            1.0,
        ),
        "atmo_height_m": float(args.atmo_height_km) * 1000.0,
        "atmo_height_exag": float(args.atmo_height_exag),
        "atmo_scatter_density": float(args.atmo_scatter_density),
        "atmo_absorption_density": float(args.atmo_absorption_density),
        "atmo_anisotropy": float(args.atmo_anisotropy),
        "atmo_scatter_color": (
            float(args.atmo_scatter_r),
            float(args.atmo_scatter_g),
            float(args.atmo_scatter_b),
            1.0,
        ),
        "atmo_absorption_color": (
            float(args.atmo_absorption_r),
            float(args.atmo_absorption_g),
            float(args.atmo_absorption_b),
            1.0,
        ),
        "nodata_value": float(args.nodata_value),
        "nodata_epsilon": float(args.nodata_epsilon),
        "nodata_fill_m": float(args.nodata_fill),
        "disp_min_m": float(args.disp_min),
        "disp_max_m": float(args.disp_max),
    }

    bpy.ops.wm.read_factory_settings(use_empty=False)
    clear_scene()

    scene = bpy.context.scene
    enable_cycles_gpu(scene)
    if args.experimental and hasattr(scene.cycles, "feature_set"):
        scene.cycles.feature_set = "EXPERIMENTAL"
    scene.cycles.preview_samples = args.viewport_samples
    scene.cycles.samples = args.render_samples
    if hasattr(scene.cycles, "preview_dicing_rate"):
        scene.cycles.preview_dicing_rate = args.viewport_dicing
    if hasattr(scene.cycles, "dicing_rate"):
        scene.cycles.dicing_rate = args.render_dicing
    if hasattr(scene.cycles, "use_adaptive_sampling"):
        scene.cycles.use_adaptive_sampling = True
    if hasattr(scene.cycles, "max_subdivisions"):
        scene.cycles.max_subdivisions = 12
    # Keep volumetrics lighter in viewport.
    if hasattr(scene.cycles, "volume_step_rate"):
        scene.cycles.volume_step_rate = 2.0
    if hasattr(scene.cycles, "volume_preview_step_rate"):
        scene.cycles.volume_preview_step_rate = 8.0

    # Camera
    bpy.ops.object.camera_add(location=(0.0, -3.4, 1.15))
    camera = bpy.context.active_object
    camera.name = "Camera"
    camera.data.lens = 85
    scene.camera = camera

    # Target + track
    bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0.0, 0.0, 0.0))
    target = bpy.context.active_object
    target.name = "Earth_Target"
    track = camera.constraints.new(type="TRACK_TO")
    track.target = target
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"

    # Sun
    bpy.ops.object.light_add(type="SUN", location=(3.0, -2.0, 2.0))
    sun = bpy.context.active_object
    sun.name = "Sun"
    sun.rotation_euler = (math.radians(50), 0.0, math.radians(120))
    sun.data.energy = 3.5
    if hasattr(sun.data, "angle"):
        sun.data.angle = math.radians(0.5)

    # World background
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (0.015, 0.02, 0.05, 1.0)
        bg.inputs[1].default_value = 1.0

    # Earth land
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=args.segments,
        ring_count=args.rings,
        radius=cfg["land_radius"],
        location=(0.0, 0.0, 0.0),
    )
    land = bpy.context.active_object
    land.name = "Earth_Land"
    bpy.ops.object.shade_smooth()

    subdiv = land.modifiers.new("Subdiv", "SUBSURF")
    subdiv.subdivision_type = "CATMULL_CLARK"
    subdiv.levels = args.subdiv_view
    subdiv.render_levels = args.subdiv_render
    if args.adaptive_subdiv and hasattr(subdiv, "use_adaptive_subdivision"):
        subdiv.use_adaptive_subdivision = True

    height_img = bpy.data.images.load(heightmap_path)
    land_mat = create_land_material(height_img, cfg)
    land.data.materials.clear()
    land.data.materials.append(land_mat)

    # Water sphere
    water_r = water_radius_for_sea(cfg, cfg["sea_level_m"])
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=args.segments,
        ring_count=args.rings,
        radius=water_r,
        location=(0.0, 0.0, 0.0),
    )
    water = bpy.context.active_object
    water.name = "Earth_Water"
    bpy.ops.object.shade_smooth()
    water_subdiv = water.modifiers.new("WaterSubdiv", "SUBSURF")
    water_subdiv.subdivision_type = "CATMULL_CLARK"
    water_subdiv.levels = args.water_subdiv_view
    water_subdiv.render_levels = args.water_subdiv_render
    water_mat = create_water_material(cfg)
    water.data.materials.clear()
    water.data.materials.append(water_mat)

    # Atmosphere shell
    if args.atmo_enable:
        atmo_r = atmosphere_radius(cfg)
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=max(128, args.segments // 2),
            ring_count=max(64, args.rings // 2),
            radius=atmo_r,
            location=(0.0, 0.0, 0.0),
        )
        atmo = bpy.context.active_object
        atmo.name = "Earth_Atmosphere"
        bpy.ops.object.shade_smooth()
        atmo_subdiv = atmo.modifiers.new("AtmoSubdiv", "SUBSURF")
        atmo_subdiv.subdivision_type = "CATMULL_CLARK"
        atmo_subdiv.levels = args.atmo_subdiv_view
        atmo_subdiv.render_levels = args.atmo_subdiv_render
        atmo_mat = create_atmosphere_material(cfg)
        atmo.data.materials.clear()
        atmo.data.materials.append(atmo_mat)
        # Render-only by default to avoid viewport lockups on heavy volume scenes.
        if args.atmo_render_only:
            atmo.hide_viewport = True
            atmo.display_type = "BOUNDS"

    # Optional animation
    if args.animate:
        scene.frame_start = args.frame_start
        scene.frame_end = args.frame_end

        sea_start = float(args.sea_level)
        sea_end = float(args.sea_end)

        r0 = water_r
        r1 = water_radius_for_sea(cfg, sea_start)
        r2 = water_radius_for_sea(cfg, sea_end)

        water.scale = (r1 / r0, r1 / r0, r1 / r0)
        water.keyframe_insert(data_path="scale", frame=args.frame_start)
        water.scale = (r2 / r0, r2 / r0, r2 / r0)
        water.keyframe_insert(data_path="scale", frame=args.frame_end)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=out_path)
    print(f"[OK] Saved blend: {out_path}")


if __name__ == "__main__":
    main()
