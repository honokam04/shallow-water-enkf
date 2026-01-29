import numpy as np
import matplotlib.pyplot as plt
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.abspath(os.path.join(current_dir, '..', 'model'))
sys.path.append(model_dir)

from shallow_water import step_forward
from prepare import pack, unpack, create_H_matrix, get_true_state
from prepare import create_observations, init_ensemble
from analysis import analyze_and_compare_tsunami
from EnKF import analysis_step
from run_enkf import run_EnKF


# CFLごとの比較実験
def main():
    try:
        eta0 = np.loadtxt("../../../data/wave_height_initial.csv",
                          delimiter=",", skiprows=1, usecols=1)
        H_minus = np.loadtxt("../../../data/bathemetry.csv",
                             delimiter=",", skiprows=1, usecols=1)

        Ls = [10, 30, 50, 60, 70, 100]
        max_heights = []
        arrival_times = []
        true_max_h = None
        true_arrival_t = None

        for L in Ls:
            print(f"dx={L / 2}...")
            stats_true, stats_ana = run_EnKF(eta0, H_minus,
                                             obs_space_interval=L,
                                             obs_time_interval_sec=3.0, N=10,
                                             inflation_factor=1.05,
                                             save_dir="../../../result/L_compare")
            max_heights.append(stats_ana['max_height'])
            arrival_times.append(stats_ana['arrival_time'])

            # 真値はどのLでも共通なので一度だけ保持
            if true_max_h is None:
                true_max_h = stats_true['max_height']
                true_arrival_t = stats_true['arrival_time']

        # グラフの作成
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # 最大波高のプロット
        ax1.plot(Ls/2, max_heights, marker='o', label='Estimated (Analysis)')
        ax1.axhline(y=true_max_h, color='r', linestyle='--',
                    label='True Value'
                    )
        ax1.set_xlabel('dx')
        ax1.set_ylabel('Max Wave Height (m)')
        ax1.set_title('dx vs Max Wave Height')
        ax1.legend()
        ax1.grid(True)

        # 到達時刻のプロット
        ax2.plot(Ls/2, arrival_times, marker='s', color='green',
                 label='Estimated (Analysis)'
                 )
        ax2.axhline(y=true_arrival_t, color='r', linestyle='--',
                    label='True Value'
                    )
        ax2.set_xlabel('dx')
        ax2.set_ylabel('Arrival Time (s)')
        ax2.set_title('dx vs Tsunami Arrival Time')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig("../../../result/L_compare/sensitivity_results.png")
        print("\nSuccess: Sensitivity graphs saved.")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
