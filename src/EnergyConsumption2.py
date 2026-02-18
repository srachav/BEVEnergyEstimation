import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

def bev_energy_model(distance, speed, elevation=None, a=None, dt_series=None, params=None, soc_init=95.0):
    """
    Compute BEV energy consumption stepwise given distance and speed arrays.
    Args:
        distance: array-like, cumulative distance [m] (monotonically increasing)
        speed: array-like, speed [m/s] (same length as distance)
        params: dict, vehicle/model parameters (optional, see defaults)
        soc_init: float, initial state of charge [%]
    Returns:
        dict with arrays for each step:
            - dt: time step at each i [s]
            - a: acceleration at each i [m/s^2]
            - delta_Ws: energy change at each i [J]
            - E_net_Ws: cumulative net energy at each i [J]
            - SOC_pct: state of charge at each i [%]
            - (plus: P_wheels_kW, P_motor_kW, P_batt_W, etc)
    """
    # --- default parameters (Nissan Leaf example) ---
    defaults = {
        "m": 1595.0,                 # kg
        "rho": 1.22563,              # kg/m^3
        "Af": 2.3316,                # m^2
        "CD": 0.28,                  # aerodynamic drag coeff
        "Cr": 1.75,                  # rolling resistance coeff
        "C1": 0.0328,                # rolling resistance base
        "C2": 4.575,                 # rolling resistance speed term
        "g": 9.8066,                 # m/s^2
        "eta_driveline": 0.92,       # driveline efficiency
        "eta_motor": 0.91,           # motor efficiency
        "eta_bat": 0.90,             # battery round-trip efficiency
        "P_aux_W": 700.0,            # auxiliary load [W]
        "alpha_rb": 0.0411,          # regen coefficient
        "eta_rb_max": 0.95,          # max regen efficiency
        "Capacity_Wh": 62000.0       # battery capacity [Wh]
    }
    if params is None:
        params = defaults.copy()
    else:
        for k, vdef in defaults.items():
            params.setdefault(k, vdef)

    distance = np.asarray(distance, dtype=float)
    speed = np.asarray(speed, dtype=float)
    n = len(speed)
    if len(distance) != n:
        raise ValueError("distance and speed must have same length")

    # dt array: either provided by caller (`dt_series`) or computed from distance/speed
    if dt_series is not None:
        dt = np.asarray(dt_series, dtype=float)
        if dt.size != n:
            raise ValueError('dt_series must have same length as distance/speed')
    else:
        # dt(i) = (distance[i] - distance[i-1]) / speed[i], dt[0]=0
        dt = np.zeros(n, dtype=float)
        dt[1:] = (distance[1:] - distance[:-1]) / np.where(speed[1:] != 0, speed[1:], 1e-6)
        dt[0] = 0.0
        # If speed at any point is very small, set dt to a fallback value
        low_speed_mask = speed < 0.01
        if np.any(low_speed_mask):
            dt[low_speed_mask] = 1

    # Acceleration: use provided `a` if given, otherwise compute from speed/distance
    if a is not None:
        a = np.asarray(a, dtype=float)
        if a.size != n:
            raise ValueError('acceleration array must match speed/distance length')
    else:
        a = np.zeros(n, dtype=float)
        # Forward difference for first point (spatial form)
        if n > 1:
            a[0] = (speed[1] - speed[0]) * speed[0] / (distance[1] - distance[0] if distance[1] != distance[0] else 1e-6)
        # Backward difference for last point
        if n > 1:
            a[-1] = (speed[-1] - speed[-2]) * speed[-1] / (distance[-1] - distance[-2] if distance[-1] != distance[-2] else 1e-6)
        # Central difference for interior points
        for i in range(1, n-1):
            denom = distance[i+1] - distance[i-1]
            a[i] = (speed[i+1] - speed[i-1]) * speed[i] / (denom if denom != 0 else 1e-6)

    # Determine road grade (theta). If elevation array is provided, compute grade
    # from elevation change over distance: grade = dh/ds, theta = arctan(grade).
    if elevation is not None:
        elev = np.asarray(elevation, dtype=float)
        if elev.size != n:
            raise ValueError('elevation must be same length as speed/distance')
        # Fit a smoothing spline (distance -> elevation) and compute its derivative
        # to obtain a smooth grade (dh/ds). Use a modest smoothing factor; if
        # spline fitting fails, fall back to finite differences.
        try:
            from scipy.interpolate import UnivariateSpline

            # heuristic smoothing factor: proportional to variance and number of points
            s = max(1.0, 0.01 * len(distance) * np.var(elev))
            spline = UnivariateSpline(distance, elev, k=3, s=s)
            # derivative of spline gives dh/ds at each distance value
            grade = spline.derivative()(distance)
            theta = np.arctan(grade)
        except Exception:
            # fallback: finite-difference grade computation (as before)
            ds = np.diff(distance, prepend=distance[0])
            with np.errstate(divide='ignore', invalid='ignore'):
                dh = np.diff(elev, prepend=elev[0])
                grade = np.zeros_like(dh)
                mask = ds != 0
                grade[mask] = dh[mask] / ds[mask]
            theta = np.arctan(grade)
    else:
        theta = np.zeros(n, dtype=float)

    # Allocate outputs
    P_wheels_kW = np.zeros(n, dtype=float)
    P_motor_kW = np.zeros(n, dtype=float)
    P_batt_W = np.zeros(n, dtype=float)
    SOC = np.zeros(n, dtype=float)
    DSOC = np.zeros(n, dtype=float)
    delta_Ws = np.zeros(n, dtype=float)
    E_net_Ws = np.zeros(n, dtype=float)

    # Unpack params
    m = params["m"]; rho = params["rho"]; Af = params["Af"]; CD = params["CD"]
    C1 = params["C1"]; C2 = params["C2"]; g = params["g"]; Cr = params["Cr"]
    eta_driveline = params["eta_driveline"]; eta_motor = params["eta_motor"]
    eta_bat = params["eta_bat"]; P_aux_W = params["P_aux_W"]
    alpha_rb = params["alpha_rb"]; eta_rb_max = params["eta_rb_max"]
    Capacity_Wh = params["Capacity_Wh"]

    for i in range(n):
        # if distance[i] == 244:
        #     print("Debug: at distance 244m, speed =", speed[i], "accel =", a[i], "grade (deg) =", np.degrees(theta[i]))
        Fi_inertia = m * a[i]
        Fi_grade = m * g * np.sin(theta[i])
        Fi_roll = m * g * np.cos(theta[i])*(C2 + C1 * speed[i]) * Cr / 1000
        Fi_aero = 0.5 * rho * Af * CD * speed[i] * speed[i]
        F_total = Fi_inertia + Fi_grade + Fi_roll + Fi_aero

        P_wheels = F_total * speed[i]  # W
        P_wheels_kW[i] = P_wheels / 1000.0

        if P_wheels >= 0.0:
            P_motor = P_wheels / (eta_driveline * eta_motor)
            P_motor_kW[i] = P_motor / 1000.0
            P_batt = (P_motor + P_aux_W) / eta_bat
        else:
            if a[i] < 0.0:
                eta_rb = np.exp(alpha_rb / np.abs(a[i]))**(- 1.0)
                eta_rb = max(0.0, min(eta_rb_max, eta_rb))
            else:
                eta_rb = 0.0
            P_motor_neg = P_wheels * eta_rb * (eta_driveline * eta_motor)
            P_motor_kW[i] = P_motor_neg / 1000.0
            P_batt = (P_motor_neg + P_aux_W) * eta_bat

        P_batt_W[i] = P_batt
        delta_Ws[i] = P_batt * dt[i]
        E_net_Ws[i] = E_net_Ws[i-1] + delta_Ws[i] if i > 0 else delta_Ws[0]
        
        DSOC[i] = P_batt / (3600 * Capacity_Wh) if i > 0 else 0
        SOC[i] = SOC[i-1] - DSOC[i] if i > 0 else soc_init - DSOC[i]
        # If SOC drops to 20% (or below), reinitialize to 95%
        if SOC[i] <= 20.0:
            SOC[i] = 95.0

    E_net_Wh = E_net_Ws / 3600.0

    return {
        "dt": dt,
        "a": a,
        "delta_Ws": delta_Ws,
        "E_net_Wh": E_net_Wh,
        "P_wheels_kW": P_wheels_kW,
        "P_motor_kW": P_motor_kW,
        "P_batt_W": P_batt_W,
        "SOC_pct": SOC,
        "cumulative_energy_Wh": E_net_Ws/ 3600,
        "EC_Wh_per_km": E_net_Wh[-1] / (distance[-1] / 1000.0) if distance[-1] > 0 else np.inf,
        "elevation": elev if 'elev' in locals() else None,
        "params_used": params
    }


def plot_power_components(distance, result, ax=None):
    """Plot wheel, motor, and battery power (kW) vs distance (km).

    Args:
        distance: array-like distance in meters (same length as result arrays)
        result: dict returned by `bev_energy_model`
        ax: optional Matplotlib Axes to plot into
    Returns:
        (fig, ax)
    """
    dist_km = np.asarray(distance, dtype=float) / 1000.0
    P_wheels = np.asarray(result.get("P_wheels_kW", np.zeros_like(dist_km)))
    P_motor = np.asarray(result.get("P_motor_kW", np.zeros_like(dist_km)))
    P_batt_kW = np.asarray(result.get("P_batt_W", np.zeros_like(dist_km))) / 1000.0

    # If elevation present in result, create a two-row figure: power on top, elevation below
    elev = result.get('elevation', None)
    if elev is not None and len(elev) == len(dist_km):
        fig, (ax_pwr, ax_elev) = plt.subplots(2, 1, sharex=True, figsize=(8, 6), gridspec_kw={'height_ratios': [3, 1]})
        ax = ax_pwr
    else:
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        else:
            fig = ax.figure

    ax.plot(dist_km, P_wheels, label="P_wheels_kW")
    ax.plot(dist_km, P_motor, label="P_motor_kW")
    ax.plot(dist_km, P_batt_kW, label="P_batt_kW")
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Power (kW)")
    ax.set_title("Power Components vs Distance")
    ax.legend(loc="best")
    ax.grid(True, which='major', linestyle='-', linewidth=0.6)
    ax.minorticks_on()
    ax.grid(True, which='minor', linestyle='--', linewidth=0.4, alpha=0.6)

    if elev is not None and len(elev) == len(dist_km):
        # plot elevation (meters) vs distance (km) on bottom subplot
        elev = np.asarray(elev, dtype=float)
        ax_elev.plot(dist_km, elev, color='tab:brown')
        ax_elev.set_xlabel('Distance (km)')
        ax_elev.set_ylabel('Elevation (m)')
        ax_elev.set_title('Elevation vs Distance')
        ax_elev.grid(True, which='major', linestyle='-', linewidth=0.6)
        ax_elev.minorticks_on()
        ax_elev.grid(True, which='minor', linestyle='--', linewidth=0.4, alpha=0.6)
        return fig, (ax, ax_elev)
    return fig, ax


def plot_soc_and_energy(distance, result):
    """Create stacked subplots: SOC (%) and E_net_Wh vs distance (km).

    Args:
        distance: array-like distance in meters
        result: dict returned by `bev_energy_model`
    Returns:
        (fig, (ax1, ax2))
    """
    dist_km = np.asarray(distance, dtype=float) / 1000.0
    soc = np.asarray(result.get("SOC_pct", np.zeros_like(dist_km)))
    E_net_Wh = np.asarray(result.get("E_net_Wh", np.zeros_like(dist_km)))

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    ax1.plot(dist_km, soc, color="tab:blue")
    ax1.set_ylabel("SOC (%)")
    ax1.set_title("State of Charge vs Distance")
    ax1.grid(True, which='major', linestyle='-', linewidth=0.6)
    ax1.minorticks_on()
    ax1.grid(True, which='minor', linestyle='--', linewidth=0.4, alpha=0.6)

    ax2.plot(dist_km, E_net_Wh, color="tab:green")
    ax2.set_ylabel("E_net (Wh)")
    ax2.set_xlabel("Distance (km)")
    ax2.set_title("Cumulative Energy vs Distance")
    ax2.grid(True, which='major', linestyle='-', linewidth=0.6)
    ax2.minorticks_on()
    ax2.grid(True, which='minor', linestyle='--', linewidth=0.4, alpha=0.6)

    fig.tight_layout()
    return fig, (ax1, ax2)

if __name__ == "__main__":
    # Try to load WLTC class 3 sheet from an Excel workbook and run model.
    # Edit this path to point to your workbook.
    example_xls = r"WLTP-DHC-12-07e.xls"
    sheet_name = 'WLTC_class_3'
    speed_col = 'WLTC class 3, version 5, vehicle speed'  # km/h
    time_col = 'Total elapsed time'  # seconds

    if os.path.exists(example_xls):
        try:
            df = pd.read_excel(example_xls, sheet_name=sheet_name)
            if speed_col in df.columns and time_col in df.columns:
                # coerce to numeric and drop rows that contain NaNs in either column
                tmp = df[[speed_col, time_col]].copy()
                tmp[speed_col] = pd.to_numeric(tmp[speed_col], errors='coerce')
                tmp[time_col] = pd.to_numeric(tmp[time_col], errors='coerce')
                tmp = tmp.dropna(subset=[speed_col, time_col])

                if tmp.empty:
                    raise ValueError('No valid rows after dropping NaNs in speed/time')

                speed_kph = tmp[speed_col].to_numpy(dtype=float)
                time_s = tmp[time_col].to_numpy(dtype=float)

                # convert km/h to m/s
                speed = speed_kph * (1000.0 / 3600.0)

                # compute dt from time column and cumulative distance
                dt = np.diff(time_s, prepend=time_s[0])
                dt[0] = 0.0
                distance = np.cumsum(speed * dt)

                # attempt to find an elevation column in the original sheet
                elev_col = None
                for c in df.columns:
                    cl = c.lower()
                    if 'elev' in cl or 'height' in cl or 'altitude' in cl:
                        elev_col = c
                        break

                if elev_col is not None:
                    elev_series = pd.to_numeric(df.loc[tmp.index, elev_col], errors='coerce')
                    # fill small gaps in elevation
                    elev_series = elev_series.ffill().bfill().fillna(0.0)
                    elevation = elev_series.to_numpy(dtype=float)
                    print(f"Using elevation column '{elev_col}' for grade computation")
                else:
                    elevation = None

                # extract acceleration column if present and align with selected rows
                accel_col = 'WLTC class 3, version 5, acceleration'
                if accel_col in df.columns:
                    acc_series = pd.to_numeric(df.loc[tmp.index, accel_col], errors='coerce')
                    acc_series = acc_series.ffill().bfill().fillna(0.0)
                    acceleration = acc_series.to_numpy(dtype=float)
                    print(f"Using acceleration column '{accel_col}'")
                else:
                    acceleration = None

                print(f"Loaded {len(distance)} samples from {example_xls} sheet '{sheet_name}'")
            else:
                print(f"Required columns not found in {example_xls}. Using synthetic example.")
                distance = np.linspace(0, 10000, 500)
                speed = 15 + 5 * np.sin(2 * np.pi * distance / 2000)
                elevation = None
                acceleration = None
        except Exception as e:
            print(f"Failed to read {example_xls}: {e}. Using synthetic example.")
            distance = np.linspace(0, 10000, 500)
            speed = 15 + 5 * np.sin(2 * np.pi * distance / 2000)
            elevation = None
            acceleration = None
    else:
        print(f"Example workbook not found at {example_xls}; using synthetic example.")
        distance = np.linspace(0, 10000, 500)
        speed = 15 + 5 * np.sin(2 * np.pi * distance / 2000)
        elevation = None
        acceleration = None

    # pass acceleration/dt if available (dt array computed above)
    try:
        result = bev_energy_model(distance, speed, elevation=elevation, a=acceleration, dt_series=dt)
    except Exception:
        result = bev_energy_model(distance, speed, elevation=elevation, a=acceleration)
    print("Total energy consumed (Wh):", result["EC_Wh_per_km"])
    plot_soc_and_energy(distance, result)
    plot_power_components(distance, result)
    plt.show()