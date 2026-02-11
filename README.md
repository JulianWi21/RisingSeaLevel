# RisingSeaLevel

Visualisierung von Meeresspiegelanstieg auf Basis von Hoehendaten (SRTM/ETOPO), mit:
- `download_data.py` fuer Laender-DEM + Wasserdaten
- `download_etopo_world.py` fuer globale ETOPO-15s-Daten
- `generate_video.py` fuer GPU-beschleunigte MP4-Renderings
- `blender_setup_world.py` / `blender_setup_world_atmo.py` fuer Blender-Weltszenen

## Ziel Fuer GitHub

Dieses Repository soll **ohne grosse Binaerdateien** gepusht werden.
Die grossen Daten werden nach dem Clone lokal per Skript heruntergeladen.

Bereits in `.gitignore` ausgeschlossen:
- `data/`, `frames/`
- `*.mp4`, `*.mov`, `*.mkv`
- `preview*.png`
- `*.blend`, `*.blend1`
- `*.npy`

## Voraussetzungen

- Windows (PowerShell) oder Linux/macOS
- Python 3.10+ (empfohlen 3.11)
- `ffmpeg` im `PATH`
- Optional, aber empfohlen: NVIDIA GPU + CUDA fuer schnelle Renderings
- Blender (fuer die `blender_setup_*` Skripte)

## 1. Projekt Klonen

```powershell
git clone <DEIN_REPO_URL>
cd RisingSeaLevel
```

## 2. Python-Umgebung

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install numpy pandas scipy pillow affine rasterio geopandas shapely torch
```

Wenn `torch` mit CUDA genutzt werden soll, ggf. passendes Wheel von der PyTorch-Seite installieren.

## 3. Laender-Daten Herunterladen (SRTM)

Beispiel Deutschland:

```powershell
python download_data.py --country "Germany" --dem-size 7680x4320 --water
```

Was passiert:
- Laendergrenze wird geladen
- SRTM-Tiles werden geladen und gemerged
- DEM wird auf Grenze geclippt
- optional resized DEM wird erzeugt
- Wasser-Raster werden vorab erstellt

## 4. Globale Weltdaten (ETOPO) Herunterladen

Fuer Welt-Rendering in Blender:

```powershell
python download_etopo_world.py --workers 6 --mem-limit 2048
```

Output (Standard):
- `data/world/etopo2022_surface_15s_world.tif`

Hinweis: grosser Download/Storage-Bedarf (mehrere GB). Plane genug freien Speicher ein.

## 5. Video Rendern (Python Pipeline)

Vorschau:

```powershell
python generate_video.py --country "Germany" --preview preview_germany.png --preview-level 25
```

Finales Video:

```powershell
python generate_video.py --country "Germany" --sea-min 0 --sea-max 200 --sea-step 0.5 --output sea_level_rise_germany.mp4
```

## 6. Welt-Szene In Blender Erzeugen (optional)

Mit Atmosphaere:

```powershell
blender --background --factory-startup --python blender_setup_world_atmo.py -- --heightmap data/world/etopo2022_surface_15s_world.tif --out data/world/world_atmo.blend --sea-level 0 --sea-end 60 --animate
```

Ohne Atmosphaere:

```powershell
blender --background --factory-startup --python blender_setup_world.py -- --heightmap data/world/etopo2022_surface_15s_world.tif --out data/world/world_basic.blend --sea-level 0 --sea-end 60 --animate
```

## 7. GitHub Copilot Ablauf (fuer deinen Bruder)

So kann er das Projekt nach dem Clone direkt mit Copilot nutzen:

1. VS Code oeffnen und Repository-Ordner laden.
2. GitHub Copilot und Copilot Chat aktivieren.
3. Im Terminal die Commands aus Abschnitt 2 und 3 oder 4 ausfuehren.
4. Copilot gezielt fragen, statt "mach alles".

Gute Copilot-Prompts:
- `Lies README.md und gib mir die naechsten 3 PowerShell-Befehle fuer einen Deutschland-Render.`
- `Erklaere mir in generate_video.py nur die Konstanten fuer Farben und UI-Balken.`
- `Passe die Wasserfarben auf dunkleres Blau an und halte den Rest unveraendert.`
- `Erhoehe die Tick-Dichte im unteren Balken auf 25m Schritte und zeige mir den Diff.`

## 8. Farben, Balken und Look einfach anpassen

Die wichtigsten Stellen sind in `generate_video.py`.

Farben:
- `OCEAN_COLOR_DEEP` = tiefer Ozean
- `OCEAN_COLOR_SHORE` = flaches Wasser/Ufer
- `RIVER_COLOR` = Fluesse
- `TERRAIN_COLORS` = Hoehen-Farbverlauf (Liste aus Hoehe + RGB)

Land-Look:
- `LAND_BRIGHTNESS`
- `LAND_SATURATION`
- Fuer ETOPO: `LAND_BRIGHTNESS_ETOPO`, `LAND_SATURATION_ETOPO`

Unterer Balken + Texte:
- Funktion `draw_text_on_frame(...)` steuert Position, Breite, Tick-Linien und Labels.
- Mit CLI kannst du Tick-Abstand direkt setzen:

```powershell
python generate_video.py --country "Germany" --ui-tick-step 25
```

Wenn `--ui-tick-step` nicht gesetzt ist, entscheidet `pick_tick_step(...)` automatisch.

Animation/Verlauf:
- `--sea-min`, `--sea-max`, `--sea-step`
- `--sea-curve linear|quadratic|log`

Beispiel (langsamer, besser lesbare Skala):

```powershell
python generate_video.py --country "Germany" --sea-min 0 --sea-max 150 --sea-step 0.25 --sea-curve linear --ui-tick-step 25 --output sea_level_rise_germany.mp4
```

## 9. Downloader + Merge (ETOPO Welt)

`download_etopo_world.py` macht:
1. Tile-Liste von NOAA laden
2. Tiles parallel herunterladen (mit Retries)
3. Alle Tiles zu einer Weltdatei mergen

Standard-Command:

```powershell
python download_etopo_world.py --workers 6 --retries 2 --mem-limit 2048
```

Nur Download (ohne Merge):

```powershell
python download_etopo_world.py --workers 6 --no-merge
```

Spaeter mergen:
- Einfach denselben Command **ohne** `--no-merge` starten.
- Bereits geladene Tiles werden uebersprungen, danach wird gemerged.

Custom Ausgabeorte:

```powershell
python download_etopo_world.py --out-dir data/world --output data/world/etopo_world.tif --workers 8 --mem-limit 4096
```

## 10. Sauber Auf GitHub Pushen (ohne grosse Dateien)

Vor Commit pruefen:

```powershell
git status
```

Falls grosse Dateien versehentlich im Index sind:

```powershell
git rm -r --cached data frames
git rm --cached *.mp4
git rm --cached preview*.png
git rm --cached *.blend
```

Dann normal committen:

```powershell
git add .
git commit -m "Add docs and source code"
git push
```
