# Project: ProjPhaseII

This repository contains tools to process a route, run vehicle simulation and LSTM inference, and compute BEV energy estimates.

This README explains how to run `ProjPhaseIIMain.py`, required Python packages, and external dependencies (Docker, Valhalla, ChromeDriver, system libs).

---

## Quick start

1. Open a terminal and change to the `src` folder:

```powershell
cd C:\HUProj\src
```

2. (Optional) Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1    # PowerShell
# or: .\.venv\Scripts\activate   # cmd.exe
```

3. Install Python dependencies (CPU TensorFlow recommended on desktop/laptop):

```powershell
pip install --upgrade pip
pip install numpy pandas scipy matplotlib requests selenium joblib tensorflow scikit-learn shapely pyproj
```

Notes:
- To avoid binary build issues with `pyproj` and `shapely`, consider using `conda` or install from the `conda-forge` channel: `conda install -c conda-forge pyproj shapely proj geos`.
- If you want a CPU-only TensorFlow, install `tensorflow-cpu` instead of `tensorflow`.

4. Ensure external services are available:

- **Docker** (if using Valhalla via Docker): install Docker Desktop (Windows). Ensure Docker is running before launching Valhalla.


 - **Valhalla routing service**: The code expects a Valhalla instance reachable at the URL configured in `projphaseii_config.json` (default local: `http://localhost:8002`). After installing Docker (see above), you can deploy the scripted Valhalla build with the following commands (example downloads Michigan extract and runs Valhalla):

```bash
docker pull ghcr.io/valhalla/valhalla-scripted:latest
mkdir -p custom_files
wget -O custom_files/michigan-latest.osm.pbf https://download.geofabrik.de/north-america/us/michigan-latest.osm.pbf
docker run -dt -v $PWD/custom_files:/custom_files -p 8002:8002 -e build_elevation=True --name valhalla_michigan ghcr.io/valhalla/valhalla-scripted:latest
```

Notes:
- The `valhalla-scripted` image will build tiles from the provided PBF in `custom_files` and start the server on port `8002`.
- For Windows PowerShell, replace `$PWD` with `$(pwd)` or `.${PWD}` when mounting, and use an appropriate `wget`/`Invoke-WebRequest` command to download the PBF.
- Building tiles can be resource- and time-consuming; ensure sufficient disk space and RAM.

See the Valhalla repository for full instructions and troubleshooting: https://github.com/valhalla/valhalla

- **Selenium / ChromeDriver**: `MapDisplayHandler3` uses Selenium to open a browser and download a GeoJSON. Install Chrome (or Chromium) and download the matching `chromedriver.exe` for your Chrome version. Put `chromedriver.exe` on your PATH or in the project folder.

  - ChromeDriver downloads: https://chromedriver.chromium.org/


5. Configure the run (optional):

- Edit `src/projphaseii_config.json` to change defaults for:
  - `map_handler` (downloadPath, driverTimeout, open_url, etc.)
  - `route_processor` (proc_coords_num, request_timeout, local_base, online_base, etc.)
  - `veh_model` (noise, gains, clip limits, stop margins, etc.)
  - `inference` (method, plotting options)
  - `bev` (vehicle parameters for `bev_energy_model`, `show_plots`)

If the config file is absent or a key is missing, the code falls back to the built-in defaults.

6. Run the main script:

```powershell
python ProjPhaseIIMain.py
```

This will:
- Load or launch a browser GUI to obtain `route.geojson` (unless already present).
- Run `RouteProcessor4` to get node attributes from Valhalla.
- Run the `vehModel` vehicle simulation (PID controller) to produce phase1 velocity samples.
- Discretize the route and run inference (LSTM) and BEV energy computation according to the config.

## Troubleshooting & notes

- If `pyproj` / `shapely` fail to install via pip on Windows, use conda (recommended):

```powershell
conda create -n projphaseii python=3.10
conda activate projphaseii
conda install -c conda-forge pyproj shapely proj geos
pip install numpy pandas scipy matplotlib requests selenium joblib tensorflow scikit-learn
```

- TensorFlow may print informational messages at import; this is normal.
- If Selenium cannot locate ChromeDriver, either put `chromedriver.exe` on the PATH or set `webdriver.Chrome(executable_path=...)` accordingly.
- Running Valhalla requires downloading and preparing OSM data and building tiles; consult Valhalla docs for production usage.

## Files of interest

- `projphaseii_config.json` — contains runtime tunables (defaults provided).
- `ProjPhaseIIMain.py` — main script to run the pipeline.
- `RouteProcessor4.py` — route processing and Valhalla client.
- `PIDController2.py` — vehicle model and simulation.
- `inferenceLSTMUseSimSpeed.py`, `inferenceLSTM2.py` — inference utilities and plotting.
- `EnergyConsumption2.py` — BEV energy model and plotting.

## Contact

If you need help running the pipeline, provide the output/error messages and I can assist further.
