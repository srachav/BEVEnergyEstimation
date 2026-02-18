import os
import time
import json
import sys
import joblib
import numpy as np
import pandas as pd
import MapDisplayHandler3
import RouteProcessor4
import PIDController2
from shapely.geometry import LineString
from shapely.ops import transform
from pyproj import Transformer, Geod
import math

# inferenceLSTM_stride contains helper functions we will reuse
import tensorflow as tf
import inferenceLSTMUseSimSpeed as ils1
import inferenceLSTM2 as ils2
# import inferenceLSTM_largerLSTMPast100 as ils3
from EnergyConsumption2 import bev_energy_model, plot_power_components, plot_soc_and_energy


class ProjPhaseIIMain:
    def __init__(self, debugMode=True):
        self.debugMode = debugMode
        self.route_geojson = "route.geojson"
        self.phase2_csv = "phase2_input.csv"
        self.nodes_csv_phase1 = "nodes_phase1.csv"
        self.nodes_csv_phase2 = "nodes_phase2.csv"
        # scaler/model defaults (adjust if needed)
        self.driving_scaler_path = r".\scalers_og\driving_scaler.pkl"
        self.road_scaler_path = r".\scalers_og\road_scaler.pkl"
        self.model_path = "best_velocity_lstm_model.keras"
        # placeholders for configs that may be populated from external config file
        self.map_handler_cfg = {}
        self.route_processor_cfg = {}
        self.veh_model_cfg = {}
        self.inference_cfg = {}
        if self.route_geojson in os.listdir(os.getcwd()) and (not self.debugMode):
            os.remove(self.route_geojson )

    def get_route_via_gui(self, handler_kwargs=None):
        # Reuse MapDisplayHandler3 GUI to obtain route.geojson in cwd
        if handler_kwargs is None:
            handler_kwargs = self.map_handler_cfg or {}
        handler = MapDisplayHandler3.MapDisplayHandler(**handler_kwargs)
        handler.initDriver()
        try:
            while True:
                handler.runMapDisplayHandler()
                if handler.userClosedBrowser:
                    break
        finally:
            try:
                handler.driver.quit()
            except Exception:
                pass
        if not os.path.exists(self.route_geojson):
            raise FileNotFoundError("route.geojson not found after GUI retrieval")
        print("route.geojson retrieved")

    def phase1_run_route_and_sim(self, rp_kwargs=None, veh_kwargs=None):
        # Load geojson coords
        with open(self.route_geojson, "r", encoding="utf-8") as f:
            gj = json.load(f)
        # Expect Feature or raw geometry
        if "features" in gj and len(gj["features"])>0:
            coords = gj["features"][0]["geometry"]["coordinates"]
        elif "geometry" in gj:
            coords = gj["geometry"]["coordinates"]
        else:
            raise RuntimeError("Cannot find coordinates in geojson")

        # coords are list of [lon, lat]
        print(f"Running RouteProcessor4 on {len(coords)} coordinates (phase1)")
        if rp_kwargs is None:
            rp_kwargs = self.route_processor_cfg or {}
        nodePD = RouteProcessor4.runRouteProcessor(coords, **rp_kwargs)
        nodePD.to_csv(self.nodes_csv_phase1, index=False)
        print(f"Saved phase1 nodes to {self.nodes_csv_phase1}")

        # Run vehicle model simulation
        if veh_kwargs is None:
            veh_kwargs = self.veh_model_cfg or {}
        veh = PIDController2.vehModel(nodePD, **veh_kwargs)
        t0 = 0
        y0 = [0, 0, 0, 0]
        h = 0.01
        print("Running vehicle simulation (phase1)")
        [tdisp, ydisp] = PIDController2.EulInt(t0, y0, h, veh.getStateDeriv, veh.getSimStopFlag)
        ydisp = np.array(ydisp)

        # ydisp columns: [distance, velocity, acceleration, jerk]
        # Resample to 1-meter distances using interpolation on distance
        dist = ydisp[:, 0]
        vel = ydisp[:, 1]
        acc = ydisp[:, 2]
        max_dist = int(np.floor(dist[-1]))
        sample_d = np.arange(0, max_dist + 1, 1)
        vel_at_m = np.interp(sample_d, dist, vel)
        acc_at_m = np.interp(sample_d, dist, acc)

        phase1_df = pd.DataFrame({
            'distance_m': sample_d,
            'velocity_mps': vel_at_m,
            'acceleration_mps2': acc_at_m
        })
        phase1_df.to_csv('phase1_velocity_at_1m.csv', index=False)
        print('Saved phase1 sampled velocity/acc to phase1_velocity_at_1m.csv')

        return phase1_df

    def discretize_geojson_1m(self, coords):
        # coords: list of [lon, lat]
        # Perform geodesic interpolation directly on the geographic coordinates
        # using pyproj.Geod to avoid projecting to a planar CRS.
        if not coords or len(coords) < 2:
            return coords

        geod = Geod(ellps='WGS84')

        # compute segment lengths and forward azimuths
        seg_lengths = []
        seg_azs = []
        for i in range(len(coords) - 1):
            lon1, lat1 = coords[i]
            lon2, lat2 = coords[i + 1]
            az12, az21, dist = geod.inv(lon1, lat1, lon2, lat2)
            seg_lengths.append(dist)
            seg_azs.append(az12)

        total_len = sum(seg_lengths)
        if total_len < 1.0:
            return coords

        # sample distances every 1 meter (including last point)
        distances = np.arange(0, total_len + 0.5, 1.0)

        # cumulative start distances for each segment
        cum = [0.0]
        for L in seg_lengths:
            cum.append(cum[-1] + L)

        pts = []
        for d in distances:
            # find segment index
            idx = int(np.searchsorted(cum, d, side='right') - 1)
            if idx < 0:
                idx = 0
            if idx >= len(seg_lengths):
                idx = len(seg_lengths) - 1

            seg_start_d = cum[idx]
            dist_into = d - seg_start_d

            lon1, lat1 = coords[idx]
            az = seg_azs[idx]
            # geod.fwd expects lon, lat, azimuth, distance (meters)
            lonp, latp, backaz = geod.fwd(lon1, lat1, az, dist_into)
            pts.append([lonp, latp])

        return pts

    def phase2_discretize_and_process(self, phase1_df):
        # read route geojson
        with open(self.route_geojson, "r", encoding="utf-8") as f:
            gj = json.load(f)
        if "features" in gj and len(gj["features"])>0:
            coords = gj["features"][0]["geometry"]["coordinates"]
        elif "geometry" in gj:
            coords = gj["geometry"]["coordinates"]
        else:
            raise RuntimeError("Cannot find coordinates in geojson")

        print("Discretizing geojson to ~1m resolution (phase2)")
        disc_coords = self.discretize_geojson_1m(coords)
        print(f"Discretized to {len(disc_coords)} coordinates")

        print("Running RouteProcessor4 on discretized coords (phase2)")
        rp_kwargs = self.route_processor_cfg or {}
        nodePD2 = RouteProcessor4.runRouteProcessor(disc_coords, **rp_kwargs)
        nodePD2.to_csv(self.nodes_csv_phase2, index=False)
        print(f"Saved phase2 nodes to {self.nodes_csv_phase2}")

        # Combine nodePD2 with phase1 velocity/acc
        nodePD2 = nodePD2.reset_index(drop=True)
        # phase1_df indexed by distance_m; if lengths differ, match by nearest distance
        phase1_df_indexed = phase1_df.set_index('distance_m')
        distances_node = None
        if 'range' in nodePD2.columns:
            distances_node = nodePD2['range'].round().astype(int).values
        else:
            distances_node = np.arange(len(nodePD2))

        vel_assign = np.zeros(len(nodePD2)) * np.nan
        acc_assign = np.zeros(len(nodePD2)) * np.nan
        max_phase1_d = phase1_df['distance_m'].max()
        for i, d in enumerate(distances_node):
            d_clamped = int(min(max(d, 0), int(max_phase1_d)))
            if d_clamped in phase1_df_indexed.index:
                vel_assign[i] = phase1_df_indexed.loc[d_clamped, 'velocity_mps']
                acc_assign[i] = phase1_df_indexed.loc[d_clamped, 'acceleration_mps2']
            else:
                # fallback interpolation
                vel_assign[i] = np.interp(d, phase1_df['distance_m'].values, phase1_df['velocity_mps'].values)
                acc_assign[i] = np.interp(d, phase1_df['distance_m'].values, phase1_df['acceleration_mps2'].values)

        nodePD2['target_speed'] = vel_assign
        nodePD2['acceleration'] = acc_assign

        # Add start_stop column: first and last entries = 1, rest = 0
        n_rows = len(nodePD2)
        start_stop = np.zeros(n_rows, dtype=int)
        if n_rows > 0:
            start_stop[0] = 1
            start_stop[-1] = 1
        nodePD2['start_stop'] = start_stop

        # Add fwd_azimuth column based on 'heading' (0..360, 0=North, 90=East,...)
        # Desired mapping: headings 0..180 -> same (0..180), headings >180 -> heading-360 (-179..-1)
        if 'heading' not in nodePD2.columns:
            nodePD2['heading'] = 0
        heading_vals = nodePD2['heading'].astype(float) % 360
        # Use pandas where to vectorize mapping
        fwd_azimuth = heading_vals.where(heading_vals <= 180, heading_vals - 360)
        nodePD2['fwd_azimuth'] = fwd_azimuth

        # Save CSV for inference script
        nodePD2.to_csv(self.phase2_csv, index=False)
        print(f"Saved combined phase2 CSV to {self.phase2_csv}")

        return self.phase2_csv

    def run_inference(self, phase2_csv, inference_cfg=None):
        """Run inference according to inference_cfg.

        Returns: out_df, horizons_unscaled, horizons_starts
        """
        if inference_cfg is None:
            inference_cfg = self.inference_cfg or {}

        use_batch_inference = inference_cfg.get('use_batch_inference', True)
        steps_to_predict_at_a_time = inference_cfg.get('steps_to_predict_at_a_time', 10)
        tau_plot = inference_cfg.get('tau_plot', 0.8)
        dt_plot = inference_cfg.get('dt_plot', 0.1)
        filter_on = inference_cfg.get('filter_on', True)

        if use_batch_inference:
            print('Running inference via inferenceLSTM_stride (stride)')
            out_df, horizons_unscaled, horizons_starts = ils1.infer_from_nodePD(
                phase2_csv, self.driving_scaler_path, self.road_scaler_path, self.model_path,
                steps_to_predict_at_a_time=steps_to_predict_at_a_time, tau_plot=tau_plot, dt_plot=dt_plot, filter_on=filter_on
            )
            ils1.plot_predictions(out_df, horizons_unscaled=horizons_unscaled, horizons_starts=horizons_starts,
                                   az_change_th=inference_cfg.get('az_change_th', 60.0), seq_alpha=inference_cfg.get('seq_alpha', 0.1))
        else:
            # fallback to sliding-window inferencer (inferenceLSTM2)
            print('Running inference via inferenceLSTM2 (sliding-window)')
            out_df, horizons_unscaled, horizons_starts = ils2.infer_from_nodePD(
                phase2_csv, self.driving_scaler_path, self.road_scaler_path, self.model_path,
                steps_to_predict_at_a_time=steps_to_predict_at_a_time, tau_plot=tau_plot, dt_plot=dt_plot, filter_on=filter_on
            )
            ils2.plot_predictions(out_df, horizons_unscaled=horizons_unscaled, horizons_starts=horizons_starts,
                                  az_change_th=inference_cfg.get('az_change_th', 60.0), seq_alpha=inference_cfg.get('seq_alpha', 0.0))

        return out_df, horizons_unscaled, horizons_starts

    def compute_and_plot_bev(self, out_df, bev_cfg=None, show_plots=True):
        """Compute BEV energy using `bev_energy_model` and plot using provided plotting functions.

        bev_cfg may provide no additional keys; behavior falls back to defaults.
        """
        bev_cfg = bev_cfg or {}
        # require columns
        if 'range' in out_df.columns and 'pred_target_speed_filtered' in out_df.columns:
            dist_arr = out_df['range'].to_numpy(dtype=float)
            speed_arr = out_df['pred_target_speed_filtered'].to_numpy(dtype=float)
            elevation_arr = out_df['elevation'].to_numpy(dtype=float) if 'elevation' in out_df.columns else None
            bev_params = bev_cfg.get('params') if bev_cfg is not None else None
            soc_init = bev_cfg.get('soc_init', 95.0) if bev_cfg is not None else 95.0
            bev_res = bev_energy_model(dist_arr, speed_arr, elevation=elevation_arr, params=bev_params, soc_init=soc_init)

            try:
                import matplotlib.pyplot as plt
                fig1, ax1 = plot_power_components(dist_arr, bev_res)
                fig1.suptitle('Power components vs distance')
                fig1.tight_layout()
                fig2, (ax_soc, ax_e) = plot_soc_and_energy(dist_arr, bev_res)
                fig2.suptitle('SOC and cumulative energy vs distance')
                fig2.tight_layout()
                if show_plots:
                    try:
                        plt.show()
                    except Exception:
                        pass
            except Exception as e:
                print('BEV plotting failed:', e)
            return bev_res
        else:
            print('Skipping BEV energy plotting: required columns missing in out_df')
            return None

  


if __name__ == '__main__':
    # Load config if available (projphaseii_config.json).
    # Use a helper that works when running from source, from a PyInstaller --onefile bundle,
    # or when a user places the config next to the running executable or in the CWD.
    def resource_path(relative_path: str) -> str:
        # 1) If running from a PyInstaller onefile bundle, files are extracted to _MEIPASS
        if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
            base = sys._MEIPASS
        # 2) If running as a frozen executable without _MEIPASS, fall back to exe dir
        elif getattr(sys, 'frozen', False):
            base = os.path.dirname(sys.executable)
        # 3) Running from source: use the script directory
        else:
            base = os.path.dirname(__file__)

        # Prefer a config present in the current working directory (user-friendly)
        cwd_path = os.path.join(os.getcwd(), relative_path)
        if os.path.exists(cwd_path):
            return cwd_path

        return os.path.join(base, relative_path)

    cfg_path = resource_path('projphaseii_config.json')
    cfg = {}
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, 'r', encoding='utf-8') as cf:
                cfg = json.load(cf)
        except Exception as e:
            print('Failed to read config file, using defaults:', e)

    # initialize main class with debugMode from config if present
    p = ProjPhaseIIMain(debugMode=cfg.get('debugMode', False))

    # populate top-level paths and model/scaler locations from config
    p.route_geojson = cfg.get('route_geojson', p.route_geojson)
    p.phase2_csv = cfg.get('phase2_csv', p.phase2_csv)
    p.nodes_csv_phase1 = cfg.get('nodes_csv_phase1', p.nodes_csv_phase1)
    p.nodes_csv_phase2 = cfg.get('nodes_csv_phase2', p.nodes_csv_phase2)
    p.driving_scaler_path = cfg.get('driving_scaler_path', p.driving_scaler_path)
    p.road_scaler_path = cfg.get('road_scaler_path', p.road_scaler_path)
    p.model_path = cfg.get('model_path', p.model_path)

    # populate component configs (use provided dicts or defaults)
    p.map_handler_cfg = cfg.get('map_handler', p.map_handler_cfg)
    p.route_processor_cfg = cfg.get('route_processor', p.route_processor_cfg)
    p.veh_model_cfg = cfg.get('veh_model', p.veh_model_cfg)
    p.inference_cfg = cfg.get('inference', p.inference_cfg)

    # Print effective configuration used for this run
    try:
        effective_cfg = {
            'debugMode': p.debugMode,
            'route_geojson': p.route_geojson,
            'phase2_csv': p.phase2_csv,
            'nodes_csv_phase1': p.nodes_csv_phase1,
            'nodes_csv_phase2': p.nodes_csv_phase2,
            'driving_scaler_path': p.driving_scaler_path,
            'road_scaler_path': p.road_scaler_path,
            'model_path': p.model_path,
            'map_handler': p.map_handler_cfg,
            'route_processor': p.route_processor_cfg,
            'veh_model': p.veh_model_cfg,
            'inference': p.inference_cfg,
            'bev': cfg.get('bev', {})
        }
        print('\nEffective configuration:')
        print(json.dumps(effective_cfg, indent=2))
    except Exception as e:
        print('Failed to print effective configuration:', e)

    # 1) get route
    if not os.path.exists(p.route_geojson):
        print('Launching GUI to retrieve route.geojson')
        try:
            p.get_route_via_gui(handler_kwargs=p.map_handler_cfg)
        except Exception as e:
            raise
    else:
        print('Found existing route.geojson')

    # 2) phase1
    phase1_df = p.phase1_run_route_and_sim()

    # 3) phase2
    phase2_csv = p.phase2_discretize_and_process(phase1_df)
    # 4) inference + BEV using helper methods (configured via projphaseii_config.json)
    try:
        out_df, horizons_unscaled, horizons_starts = p.run_inference(phase2_csv, inference_cfg=p.inference_cfg)
        # Save predictions back into phase2 CSV (overwrite)
        out_df.to_csv(phase2_csv, index=False)
        print(f'Wrote predictions into {phase2_csv}')

        # Compute BEV energy and plot using config
        bev_cfg = cfg.get('bev', {})
        # pass vehicle/model params into bev_energy_model via bev_cfg['params'] inside compute_and_plot_bev
        p.compute_and_plot_bev(out_df, bev_cfg=bev_cfg, show_plots=bev_cfg.get('show_plots', True))
    except Exception as e:
        print('Inference/BEV failed:', e)
    print('Done')
