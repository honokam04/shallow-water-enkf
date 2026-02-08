import numpy as np
import matplotlib.pyplot as plt
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.abspath(os.path.join(current_dir, '..', 'model'))
sys.path.append(model_dir)

from analysis import analyze_and_compare_tsunami
from run_realtime import run_EnKF_real


def plot_forecast_at_coast():
    # データ読み込み
    eta0 = np.loadtxt("../../../data/wave_height_initial.csv",
                      delimiter=",", skiprows=1, usecols=1
                      )
    H_minus = np.loadtxt("../../../data/bathemetry.csv",
                         delimiter=",", skiprows=1, usecols=1
                         )

    # 比較したい同化終了時間（秒）
    end_times = [150, 300, 450, 600, 750]
    colors = plt.cm.viridis(np.linspace(0, 1, len(end_times)))

    plt.figure(figsize=(10, 8))

    true_plotted = False
    N = 75

    for i, t_end in enumerate(end_times):
        print(f"Running simulation for assimilation_end_time = {t_end}s...")

        # EnKF実行
        eta_coast, eta_true_coast, dt = run_EnKF_real(
            eta0, H_minus,
            obs_space_interval=60,
            obs_time_interval_sec=3.0,
            N=N,
            total_time=3000.0,
            assimilation_end_time=t_end,
            save_dir=f"../../../result_rk4/forecast_{t_end}"
        )

        stats_true, stats_ana = analyze_and_compare_tsunami(
            eta_true_coast[:, np.newaxis],
            eta_coast[:, np.newaxis],
            dt, threshold=0.2
        )

        t_arrival_true = stats_true['arrival_time']
        t_arrival_ana = stats_ana['arrival_time']
        t_max_true = stats_true['max_time']
        t_max_ana = stats_ana['max_time']
        h_max_true = stats_true['max_height']
        h_max_ana = stats_ana['max_height']

        nt = len(eta_coast)
        time_axis = np.arange(nt) * dt
        offset = i * 4.0

        # 各段に真値の点線をプロット
        plt.plot(time_axis, eta_true_coast + offset, color='black',
                 linestyle='--', alpha=0.6, label='True' if i==0 else ""
                 )

        split_idx = int(t_end / dt)
        # 同化期間
        plt.plot(time_axis[:split_idx], eta_coast[:split_idx] + offset,
                 color='blue', linewidth=2, label='EnKF' if i==0 else ""
                 )
        # 予測期間
        plt.plot(time_axis[split_idx:], eta_coast[split_idx:] + offset,
                 color='red', linewidth=1.5, label='Forecast' if i==0 else ""
                 )

        # 第一波到達時刻
        if t_arrival_true is not None:
            plt.vlines(t_arrival_true, offset - 0.5, offset + 1.5,
                       color='green', linestyle=':', linewidth=1,
                       label='True Arrival' if i==0 else ""
                       )
        if t_arrival_ana is not None:
            plt.vlines(t_arrival_ana, offset - 0.5, offset + 1.5,
                       color='green', linestyle='-', linewidth=2,
                       label='Est. Arrival' if i==0 else ""
                       )

        # 最大波到達時刻
        if t_max_true is not None:
            plt.vlines(t_max_true, offset + h_max_true - 4.0,
                       offset + h_max_true + 1.0,
                       color='orange', linestyle=':', linewidth=1,
                       label='True Peak' if i==0 else ""
                       )
        if t_max_ana is not None:
            plt.vlines(t_max_ana, offset + h_max_ana - 1.0,
                       offset + h_max_ana + 1.0,
                       color='orange', linestyle='-', linewidth=2,
                       label='Est. Peak' if i==0 else ""
                       )

        # ラベル付け
        plt.text(-150, offset, f"{t_end} s",
                 verticalalignment='center', fontweight='bold'
                 )

    plt.xlabel("Time (s)")
    plt.ylabel("Wave Height at Coast")
    plt.title("Forecast at Coast for different Assimilation End Times")
    plt.grid(axis='x', alpha=0.3)
    plt.legend(loc='upper right')

    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.yticks([])

    plt.tight_layout()
    plt.savefig(f"../../../result_rk4/forecast_at_coast_comparison_N={N}.png")
    plt.show()


if __name__ == "__main__":
    plot_forecast_at_coast()
