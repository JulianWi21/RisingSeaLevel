# RisingSeaLevel (ETOPO-only)

Dieses Repo ist auf einen klaren Workflow reduziert:
- ETOPO 2022 Tiles downloaden
- zu einer Welt-GeoTIFF mergen
- mit `generate_video.py` ein Sea-Level-Video rendern

Blender-Skripte sind absichtlich aus dem Git-Repo ausgeschlossen.

## Was ist im Repo

- `download_etopo_world.py` -> Download + Merge von ETOPO 15s
- `generate_video.py` -> Video-Rendering (CPU/GPU)

## Was ist ausgeschlossen (wichtig fuer GitHub)

Grossdateien und lokale Outputs werden nicht eingecheckt:
- `data/`
- `frames/`
- `*.mp4`, `*.mov`, `*.mkv`
- `preview*.png`
- `*.npy`
- `*.blend`, `*.blend1`
- `blender_setup_world.py`
- `blender_setup_world_atmo.py`

## Voraussetzungen

- Python 3.10+ (empfohlen 3.11)
- `ffmpeg` im PATH
- Optional NVIDIA GPU + CUDA (sonst CPU)

## 1. Clone + Setup

```powershell
git clone https://github.com/GalusPeres/RisingSeaLevel.git
cd RisingSeaLevel
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install numpy pandas scipy pillow affine rasterio geopandas shapely torch
```

## 2. ETOPO downloaden und mergen

Standard:

```powershell
python download_etopo_world.py --workers 6 --retries 2 --mem-limit 2048
```

Output:
- `data/world/etopo2022_surface_15s_world.tif`

Nur Download ohne Merge:

```powershell
python download_etopo_world.py --workers 6 --no-merge
```

Spaeter nur Merge:
- denselben Command erneut ohne `--no-merge` ausfuehren
- vorhandene Tiles werden uebersprungen

## 3. Video aus ETOPO erzeugen

Schnelle Vorschau:

```powershell
python generate_video.py --dem data/world/etopo2022_surface_15s_world.tif --country "World" --preview preview_world.png --preview-level 25
```

Finales Video:

```powershell
python generate_video.py --dem data/world/etopo2022_surface_15s_world.tif --country "World" --sea-min 0 --sea-max 200 --sea-step 1.0 --sea-curve linear --ui-tick-step 25 --output sea_level_rise_world_etopo.mp4
```

## Farben und unterer Balken anpassen

Datei: `generate_video.py`

Wichtige Farbwerte:
- `OCEAN_COLOR_DEEP`
- `OCEAN_COLOR_SHORE`
- `RIVER_COLOR`
- `TERRAIN_COLORS`

Land-Look:
- `LAND_BRIGHTNESS_ETOPO`
- `LAND_SATURATION_ETOPO`

Unterer Balken/Ticks/Text:
- Funktion `draw_text_on_frame(...)`
- Tick-Abstand direkt per CLI:

```powershell
python generate_video.py --dem data/world/etopo2022_surface_15s_world.tif --country "World" --ui-tick-step 25 --output sea_level_rise_world_etopo.mp4
```

## Fuer schwache Laptops

Erst nur Preview rendern:

```powershell
python generate_video.py --dem data/world/etopo2022_surface_15s_world.tif --country "World" --preview preview_world.png --preview-level 20
```

Dann kurzes Testvideo:

```powershell
python generate_video.py --dem data/world/etopo2022_surface_15s_world.tif --country "World" --sea-min 0 --sea-max 80 --sea-step 2.0 --output world_test.mp4
```

## GitHub Copilot Prompt fuer deinen Bruder

```text
Lies README.md und gib mir die naechsten 3 PowerShell-Befehle fuer ETOPO-Download, Merge und einen schnellen Preview-Render.
```

## Falls Blender-Dateien schon getrackt sind

Einmalig aus Git entfernen:

```powershell
git rm --cached blender_setup_world.py blender_setup_world_atmo.py
git commit -m "Stop tracking Blender setup scripts"
git push
```
