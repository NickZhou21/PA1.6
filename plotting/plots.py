import os
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np

def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def plot_timeseries(log: Dict[str, List[float]], out_path: str):
    ''' Plot the time series of temperatures and thresholds. 
        log: A dictionary containing time series data with keys:
            - "t": time points
            - "T_true": true temperature
            - "T_meas": measured/filtered temperature
            - "T_out": outdoor temperature
            - "lower": lower threshold (optional value)
            - "upper": upper threshold (optional value)
            - "setpoint": setpoint temperature
        out_path: Path to save the output plot image.
    '''
    _ensure_dir(out_path)
    t = log["t"]
    T_true = log["T_true"]
    T_meas = log["T_meas"]
    T_out = log["T_out"]
    lower = log["lower"]
    upper = log["upper"]
    setpoint = log["setpoint"]
    fig, ax = plt.subplots()
    ax.plot(t, T_true, label="True T")
    ax.plot(t, T_meas, label="Measured/Filtered T")
    ax.plot(t, setpoint, label="Setpoint")
    ax.plot(t, T_out, label="Outdoor T", linestyle="--")
    if lower and upper:
        ax.plot(t, lower, linestyle=":", label="Lower threshold")
        ax.plot(t, upper, linestyle=":", label="Upper threshold")
        ax.fill_between(t, lower, upper, color="gray", alpha=0.2, label="Deadband region")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Temperatures vs Time")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def plot_heater(log: Dict[str, List[float]], out_path: str):
    ''' Plot the heater state over time.
        log: A dictionary containing time series data with keys:
            - "t": time points
            - "heater": heater state (0=OFF, 1=ON)
        out_path: Path to save the output plot image.
    '''
    _ensure_dir(out_path)
    t = log["t"]
    heater = log["heater"]
    fig, ax = plt.subplots()
    ax.step(t, heater, where="post")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Heater (0=OFF, 1=ON)")
    ax.set_ylim(-0.1, 1.1)
    ax.set_title("Heater State Timeline")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def plot_error(log: Dict[str, List[float]], out_path: str):
    ''' Plot the tracking error (setpoint - measured temperature) over time.
        log: A dictionary containing time series data with keys:
            - "t": time points
            - "error": tracking error (setpoint - measured temperature)
        out_path: Path to save the output plot image.
    '''
    _ensure_dir(out_path)
    t = log["t"]
    error = log["error"]
    fig, ax = plt.subplots()
    ax.plot(t, error)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Error (setpoint - measured)")
    ax.set_title("Tracking Error")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def plot_duty(log: Dict[str, List[float]], out_path: str, window:int=60):
    ''' Plot the rolling duty cycle of the heater over time.
        log: A dictionary containing time series data with keys:
            - "t": time points
            - "heater": heater state (0=OFF, 1=ON)
        out_path: Path to save the output plot image.
        window: Size of the rolling window in seconds for duty cycle calculation.
    '''
    _ensure_dir(out_path)
    heater = log["heater"]
    t = log["t"]
    duty = []
    s = 0.0
    buf = []
    for b in heater:
        buf.append(b)
        s += b
        if len(buf) > window:
            s -= buf.pop(0)
        duty.append(s / len(buf))
    fig, ax = plt.subplots()
    ax.plot(t, duty)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"Heater Duty (rolling, window={window})")
    ax.set_title("Heater Duty Cycle (Rolling)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def plot_predictive(log: Dict[str, List[float]], out_path: str):
    ''' Plot the predictive control diagnostics including measured temperature,
        predicted temperature, and thresholds over time.
        log: A dictionary containing time series data with keys:
            - "t": time points
            - "T_meas": measured/filtered temperature
            - "T_pred": predicted temperature (optional)
            - "lower": lower threshold (optional)
            - "upper": upper threshold (optional)
        out_path: Path to save the output plot image.
    '''
    _ensure_dir(out_path)
    t = log["t"]
    T_pred = log.get("T_pred", [])
    lower = log.get("lower", [])
    upper = log.get("upper", [])
    T_meas = log["T_meas"]
    fig, ax = plt.subplots()
    ax.plot(t, T_meas, label="Measured/Filtered T")
    if T_pred:
        ax.plot(t, T_pred, label="Predicted T (lookahead)")
    if lower and upper:
        ax.plot(t, lower, linestyle=":", label="Lower threshold")
        ax.plot(t, upper, linestyle=":", label="Upper threshold")
        ax.fill_between(t, lower, upper, alpha=0.1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Predictive Control Diagnostics")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_monte_carlo(
    times: List[float],
    temperature_runs: List[List[float]],
    heater_duty_pct: List[float],
    rmse_values: List[float],
    setpoint: float,
    deadband: float,
    out_path: str,
):
    _ensure_dir(out_path)
    temperatures = np.asarray(temperature_runs)
    percentile_10, median, percentile_90 = np.percentile(temperatures, [10, 50, 90], axis=0)

    fig = plt.figure(figsize=(11, 8))
    grid = fig.add_gridspec(2, 2, height_ratios=[2.0, 1.0])
    temperature_ax = fig.add_subplot(grid[0, :])
    duty_ax = fig.add_subplot(grid[1, 0])
    rmse_ax = fig.add_subplot(grid[1, 1])

    for index, run in enumerate(temperature_runs):
        temperature_ax.plot(
            times,
            run,
            color="#73858c",
            alpha=0.18,
            linewidth=0.7,
            label="Individual runs" if index == 0 else None,
        )
    temperature_ax.fill_between(
        times,
        percentile_10,
        percentile_90,
        color="#4f9b91",
        alpha=0.28,
        label="10th-90th percentile",
    )
    temperature_ax.plot(times, median, color="#176d65", linewidth=2, label="Median")
    temperature_ax.axhline(setpoint, color="#c5683b", linewidth=1.5, label="Setpoint")
    temperature_ax.axhline(setpoint - deadband / 2.0, color="#777777", linestyle="--", linewidth=1)
    temperature_ax.axhline(setpoint + deadband / 2.0, color="#777777", linestyle="--", linewidth=1,
                           label="Deadband limits")
    temperature_ax.set_title(f"Temperature Across {len(temperature_runs)} Random Seeds")
    temperature_ax.set_xlabel("Time (s)")
    temperature_ax.set_ylabel("True room temperature (°C)")
    temperature_ax.legend(loc="best", ncol=3)
    temperature_ax.grid(axis="y", alpha=0.2)

    for axis, values, label, color in (
        (duty_ax, heater_duty_pct, "Heater duty (%)", "#4f9b91"),
        (rmse_ax, rmse_values, "Temperature RMSE (°C)", "#c5683b"),
    ):
        axis.boxplot(values, showfliers=False, widths=0.35)
        point_positions = np.linspace(0.88, 1.12, len(values))
        axis.scatter(point_positions, values, color=color, alpha=0.75, s=24, zorder=3)
        axis.set_xlim(0.7, 1.3)
        axis.set_xticks([1], [f"{len(values)} runs"])
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.2)

    duty_ax.set_title("Heater Use")
    rmse_ax.set_title("Regulation Error")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
