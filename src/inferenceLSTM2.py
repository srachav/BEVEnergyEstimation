import os
import math
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Feature lists (kept small and compatible with other modules)
DRIVING_FEATURES = ['target_speed', 'acceleration']
ROAD_FEATURES = [
    'elevation', 'fwd_azimuth', 'speed_limit_mps', 'edge_speed_mps',
    'edge_curvature', 'traffic_signal', 'stop_sign', 'yield_sign', 'round_about',
    'edge_classification', 'edge_link', 'start_stop'
]

DRIVING_FEATURES_TO_SCALE = ['target_speed', 'acceleration']
ROAD_FEATURES_TO_SCALE = ['elevation', 'fwd_azimuth', 'speed_limit_mps', 'edge_speed_mps', 'edge_curvature']
TARGET_FEATURES_PREDICTED = ['target_speed', 'acceleration']

classification_map = {
    'motorway': 0,
    'trunk': 1,
    'primary': 2,
    'secondary': 3,
    'tertiary': 4,
    'unclassified': 5,
    'residential': 6,
    'service_other': 7
}


def _ensure_and_prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in set(DRIVING_FEATURES + ROAD_FEATURES + ['heading', 'range']):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    if 'edge_classification' in df.columns:
        text_vals = df['edge_classification'].astype(str).str.strip().str.lower()
        text_vals = text_vals.replace({'service other': 'service_other', 'serviceother': 'service_other'})
        mapped = text_vals.map(classification_map)
        df['edge_classification'] = mapped.fillna(5).astype(int)

    for c in DRIVING_FEATURES + ROAD_FEATURES:
        if c not in df.columns:
            df[c] = 0

    return df


def infer_from_nodePD(nodePD, driving_scaler_path, road_scaler_path, model_path,
                      past_steps=50, future_steps=100, steps_to_predict_at_a_time=10,
                      filter_on=True, tau_plot=0.7, dt_plot=0.1):
    """Sliding-window multi-step inference. Returns (out_df, horizons_unscaled, horizons_starts).

    - out_df: original dataframe with `pred_target_speed` and `pred_acceleration` columns added.
    - horizons_unscaled: list of inverse-transformed predicted horizons (ndarray each shape (future_steps, n_targets)).
    - horizons_starts: list of start indices (in original df) for each horizon.
    """
    if isinstance(nodePD, str):
        raw_df = pd.read_csv(nodePD, low_memory=False)
    else:
        raw_df = nodePD.copy()

    raw_df = _ensure_and_prepare_df(raw_df)

    driving_scaler = joblib.load(driving_scaler_path)
    road_scaler = joblib.load(road_scaler_path)
    model = tf.keras.models.load_model(model_path)

    # scale
    scaled_df = raw_df.copy()
    if DRIVING_FEATURES_TO_SCALE:
        scaled_df[DRIVING_FEATURES_TO_SCALE] = driving_scaler.transform(scaled_df[DRIVING_FEATURES_TO_SCALE])
    if ROAD_FEATURES_TO_SCALE:
        # ensure float dtype to avoid pandas assignment errors
        scaled_df[ROAD_FEATURES_TO_SCALE] = scaled_df[ROAD_FEATURES_TO_SCALE].astype(float)
        scaled_df[ROAD_FEATURES_TO_SCALE] = road_scaler.transform(scaled_df[ROAD_FEATURES_TO_SCALE])

    # pad at both ends so windows exist for the entire original sequence
    pre_pad = pd.concat([scaled_df.iloc[[0]]] * past_steps, ignore_index=True)
    post_pad = pd.concat([scaled_df.iloc[[-1]]] * future_steps, ignore_index=True)
    padded = pd.concat([pre_pad, scaled_df, post_pad], ignore_index=True)

    # initialize windows
    current_past_driving = padded[DRIVING_FEATURES].iloc[0:past_steps].values.copy()
    current_past_road = padded[ROAD_FEATURES].iloc[0:past_steps].values.copy()
    future_road_df = padded[ROAD_FEATURES]

    current_idx = past_steps
    prediction_horizon = len(raw_df)

    all_pred_scaled = []
    horizons_scaled = []
    horizons_starts = []

    while len(all_pred_scaled) < prediction_horizon:
        future_start = current_idx
        future_end = future_start + future_steps
        if future_end > len(future_road_df):
            break
        input_future_road = future_road_df.iloc[future_start:future_end].values

        x_past_d = np.expand_dims(current_past_driving, axis=0)
        x_past_r = np.expand_dims(current_past_road, axis=0)
        x_future_r = np.expand_dims(input_future_road, axis=0)

        pred_future_scaled = model.predict([x_past_d, x_past_r, x_future_r], verbose=0)[0]

        # store full horizon for overlay plotting
        horizons_scaled.append(pred_future_scaled.copy())
        start_orig = current_idx - past_steps
        horizons_starts.append(start_orig)

        # take the first steps_to_predict_at_a_time to advance
        new_preds = pred_future_scaled[:steps_to_predict_at_a_time]

        # update past driving window
        current_past_driving = np.concatenate((current_past_driving[steps_to_predict_at_a_time:], new_preds), axis=0)

        # update past road window using actual road features
        new_road_start = current_idx
        new_road_end = current_idx + steps_to_predict_at_a_time
        if new_road_end > len(future_road_df):
            break
        new_road = future_road_df.iloc[new_road_start:new_road_end].values
        current_past_road = np.concatenate((current_past_road[steps_to_predict_at_a_time:], new_road), axis=0)

        # append new_preds to aggregated list
        all_pred_scaled.extend(new_preds.tolist())

        current_idx += steps_to_predict_at_a_time

    # trim
    if len(all_pred_scaled) > prediction_horizon:
        all_pred_scaled = all_pred_scaled[:prediction_horizon]

    all_pred_scaled = np.array(all_pred_scaled)

    # inverse transform
    if all_pred_scaled.size == 0:
        inv_all = np.zeros((0, len(TARGET_FEATURES_PREDICTED)))
    else:
        inv_all = driving_scaler.inverse_transform(all_pred_scaled)

    # If necessary, pad inv_all so it matches prediction_horizon
    if inv_all.shape[0] < prediction_horizon:
        pad = np.full((prediction_horizon - inv_all.shape[0], inv_all.shape[1]), np.nan)
        inv_all = np.vstack([inv_all, pad])

    # Apply lookahead-based zeroing on the predicted speeds (before filtering)
    target_idx = TARGET_FEATURES_PREDICTED.index('target_speed')
    n_pred = inv_all.shape[0]
    pred_full = inv_all[:, target_idx].copy()

    # helper to get column values aligned to prediction horizon
    def _col_n(col):
        if col in raw_df.columns:
            return raw_df[col].iloc[0:prediction_horizon].values[:n_pred]
        return np.zeros(n_pred)

    # Simplified zeroing rule: if the (simulated) target speed is < 3 m/s
    # and the predicted speed is < 4 m/s, set prediction to zero.
    target_speed_arr = _col_n('target_speed')
    for i in range(1, n_pred):
        if np.isnan(pred_full[i]):
            continue
        if pred_full[i] < 5.0 and (target_speed_arr[i] < 4.0):
            pred_full[i] = 0.0

    # write back adjusted speeds into inv_all
    inv_all[:, target_idx] = pred_full

    # build out_df
    out_df = raw_df.copy()
    out_df['pred_target_speed'] = inv_all[:, TARGET_FEATURES_PREDICTED.index('target_speed')]
    out_df['pred_acceleration'] = inv_all[:, TARGET_FEATURES_PREDICTED.index('acceleration')]

    # Store raw predicted speed before optional filtering
    out_df['pred_target_speed_raw'] = out_df['pred_target_speed'].to_numpy(dtype=float)

    # Optionally filter predicted speed (zero-phase lowpass) and store filtered result
    if filter_on:
        pred_full = out_df['pred_target_speed_raw'].to_numpy(dtype=float)
        s = pd.Series(pred_full)
        s = s.ffill().bfill().fillna(0.0)
        pred_filled = s.to_numpy(dtype=float)
        fc_plot = 1.0 / (2.0 * math.pi * float(tau_plot))
        fs_plot = 1.0 / float(dt_plot)
        try:
            b_plot, a_plot = butter(1, fc_plot, fs=fs_plot, btype='low', analog=False)
        except TypeError:
            Wn_plot = fc_plot / (0.5 * fs_plot)
            b_plot, a_plot = butter(1, Wn_plot, btype='low')
        try:
            pred_filtered = filtfilt(b_plot, a_plot, pred_filled)
        except Exception:
            pred_filtered = pred_filled
        out_df['pred_target_speed_filtered'] = pred_filtered
        # Make the primary prediction column be the filtered values for downstream usage
        out_df['pred_target_speed'] = out_df['pred_target_speed_filtered']
    else:
        out_df['pred_target_speed_filtered'] = out_df['pred_target_speed_raw']

    # inverse-transform horizons for plotting
    horizons_unscaled = []
    for h in horizons_scaled:
        try:
            horizons_unscaled.append(driving_scaler.inverse_transform(h))
        except Exception:
            horizons_unscaled.append(np.full_like(h, np.nan))

    return out_df, horizons_unscaled, horizons_starts


def plot_predictions(out_df, horizons_unscaled=None, horizons_starts=None, az_change_th=30.0, seq_alpha=0.08):
    """Plot predicted vs actual speeds and accelerations with discrete condition subplot.
    If `horizons_unscaled` and `horizons_starts` are provided, overlay the predicted horizons.
    Note: filtering is applied in `infer_from_nodePD` and stored in `out_df['pred_target_speed']`.
    """
    n = len(out_df)
    if n == 0:
        raise ValueError('out_df is empty')

    x = out_df['range'].values if 'range' in out_df.columns else np.arange(n)
    # convert x (assumed meters) to distance in kilometers for plotting
    x_km = x / 1000.0

    pred_speed = out_df['pred_target_speed'].to_numpy(dtype=float)
    actual_speed = out_df['target_speed'].to_numpy(dtype=float) if 'target_speed' in out_df.columns else np.full(n, np.nan)
    edge_speed = out_df['edge_speed_mps'].to_numpy(dtype=float) if 'edge_speed_mps' in out_df.columns else np.full(n, np.nan)
    speed_limit = out_df['speed_limit_mps'].to_numpy(dtype=float) if 'speed_limit_mps' in out_df.columns else np.full(n, np.nan)
    # convert to mph for plotting
    mph = 2.2369362920544
    pred_speed_mph = pred_speed * mph
    actual_speed_mph = actual_speed * mph
    edge_speed_mph = edge_speed * mph
    speed_limit_mph = speed_limit * mph

    # azimuth change
    if 'fwd_azimuth' in out_df.columns:
        az = np.asarray(out_df['fwd_azimuth'].astype(float).fillna(0.0))
        az = ((az + 180.0) % 360.0) - 180.0
        az_diff = np.zeros(n)
        az_diff[1:] = np.abs(((az[1:] - az[:-1] + 180.0) % 360.0) - 180.0)
        az_change = az_diff > az_change_th
    else:
        az_change = np.zeros(n, dtype=bool)

    def col_bool(name):
        return (out_df[name].fillna(0).astype(float).to_numpy() > 0) if name in out_df.columns else np.zeros(n, dtype=bool)

    traffic_signal = col_bool('traffic_signal')
    start_stop = col_bool('start_stop')
    yield_sign = col_bool('yield_sign')
    stop_sign = col_bool('stop_sign')
    round_about = col_bool('round_about')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle('Speeds vs Distance (km)')
    # Plot predicted speed (in mph)
    ax1.plot(x_km, pred_speed_mph, label='Predicted Speed', color='tab:blue')
    # if not np.all(np.isnan(actual_speed_mph)):
    #     ax1.plot(x_km, actual_speed_mph, label='Target/Actual Speed', color='tab:red', linestyle='--')
    if not np.all(np.isnan(edge_speed_mph)):
        ax1.plot(x_km, edge_speed_mph, label='Edge Speed', color='tab:green', linestyle='-.')
    if not np.all(np.isnan(speed_limit_mph)):
        ax1.plot(x_km, speed_limit_mph, label='Speed Limit', color='tab:orange', linestyle=':')
    ax1.set_ylabel('Speed (mph)')
    ax1.legend()
    ax1.grid(True)
    ax1.minorticks_on()
    ax1.grid(which='minor', linestyle='--', linewidth=0.4, alpha=0.6)

    def plot_row(cond, y, label, color):
        ax2.scatter(x_km[cond], np.full(np.count_nonzero(cond), y), marker='s', color=color, s=8, label=label)
    ax2.minorticks_on()
    ax2.grid(which='minor', linestyle='--', linewidth=0.4, alpha=0.6)

    plot_row(traffic_signal, 5, 'traffic_signal', 'purple')
    plot_row(start_stop, 4, 'start_stop', 'black')
    plot_row(yield_sign, 3, 'yield_sign', 'brown')
    plot_row(stop_sign, 2, 'stop_sign', 'orange')
    plot_row(round_about, 1, 'round_about', 'cyan')
    plot_row(az_change, 0, f'az_change>{az_change_th}deg', 'magenta')

    ax2.set_yticks([0, 1, 2, 3, 4, 5])
    ax2.set_yticklabels([f'az>{az_change_th}', 'round_about', 'stop_sign', 'yield_sign', 'start_stop', 'traffic_signal'])
    ax2.set_xlabel('Distance (km)')
    ax2.set_ylim(-0.5, 5.5)
    ax2.grid(False)
    ax2.legend(loc='upper right')

    # overlay horizons if present
    if horizons_unscaled is not None and horizons_starts is not None:
        try:
            for h, st in zip(horizons_unscaled, horizons_starts):
                if h is None:
                    continue
                pred_speed_seq = h[:, TARGET_FEATURES_PREDICTED.index('target_speed')]
                # drop NaNs
                valid_mask = np.isfinite(pred_speed_seq)
                if not np.any(valid_mask):
                    continue
                pred_speed_seq = pred_speed_seq[valid_mask]
                x_seq = x[st: st + len(pred_speed_seq)] if len(x) > st else np.arange(st, st + len(pred_speed_seq))
                x_seq = x_seq / 1000.0
                if len(x_seq) < len(pred_speed_seq):
                    pred_speed_seq = pred_speed_seq[:len(x_seq)]
                ax1.plot(x_seq, pred_speed_seq * mph, color='blue', alpha=seq_alpha, linewidth=0.7)
        except Exception:
            pass

    plt.tight_layout()
    fig.subplots_adjust(top=0.92)
    plt.show()

    # acceleration plot: compute acceleration from pred_target_speed via spatial derivative
    speed = pred_speed
    distance = x
    a = np.full(n, np.nan, dtype=float)
    if n > 1:
        # forward difference for first point (spatial form)
        denom = (distance[1] - distance[0]) if distance[1] != distance[0] else 1e-6
        a[0] = (speed[1] - speed[0]) * speed[0] / denom
        # backward difference for last point
        denom = (distance[-1] - distance[-2]) if distance[-1] != distance[-2] else 1e-6
        a[-1] = (speed[-1] - speed[-2]) * speed[-1] / denom
        # central difference for interior points
        for i in range(1, n-1):
            denom = distance[i+1] - distance[i-1]
            denom = denom if denom != 0 else 1e-6
            a[i] = (speed[i+1] - speed[i-1]) * speed[i] / denom

    actual_acc = out_df['acceleration'].to_numpy(dtype=float) if 'acceleration' in out_df.columns else np.full(n, np.nan)

    # create two-row acceleration plot: top = acceleration traces, bottom = discrete conditions
    figa, (axa1, axa2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    figa.suptitle('Acceleration vs Distance (km)')
    axa1.plot(x_km, a, label='Predicted Acceleration (from pred_target_speed)', color='tab:blue')
    # if not np.all(np.isnan(actual_acc)):
    #     axa1.plot(x_km, actual_acc, label='Actual Acceleration', color='tab:red', linestyle='--')
    axa1.set_ylabel('Acceleration (m/s^2)')
    axa1.legend()
    axa1.grid(True)
    axa1.minorticks_on()
    axa1.grid(which='minor', linestyle='--', linewidth=0.4, alpha=0.6)

    # reuse condition booleans from speed plot to annotate acceleration subplot
    def plot_row_acc(cond, y, label, color):
        axa2.scatter(x_km[cond], np.full(np.count_nonzero(cond), y), marker='s', color=color, s=8, label=label)

    plot_row_acc(traffic_signal, 5, 'traffic_signal', 'purple')
    plot_row_acc(start_stop, 4, 'start_stop', 'black')
    plot_row_acc(yield_sign, 3, 'yield_sign', 'brown')
    plot_row_acc(stop_sign, 2, 'stop_sign', 'orange')
    plot_row_acc(round_about, 1, 'round_about', 'cyan')
    plot_row_acc(az_change, 0, f'az_change>{az_change_th}deg', 'magenta')
    
    axa2.set_yticks([0, 1, 2, 3, 4, 5])
    axa2.set_yticklabels([f'az>{az_change_th}', 'round_about', 'stop_sign', 'yield_sign', 'start_stop', 'traffic_signal'])
    axa2.set_xlabel('Distance (km)')
    axa2.set_ylim(-0.5, 5.5)
    axa2.grid(False)
    axa2.minorticks_on()
    axa2.grid(which='minor', linestyle='--', linewidth=0.4, alpha=0.6)
    axa2.legend(loc='upper right')

    plt.tight_layout()
    figa.subplots_adjust(top=0.90)
    plt.show()


if __name__ == '__main__':
    print('inferenceLSTM2 module loaded - use infer_from_nodePD(...)')
