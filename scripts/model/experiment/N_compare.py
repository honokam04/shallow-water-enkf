import numpy as np
import matplotlib.pyplot as plt
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.abspath(os.path.join(current_dir, '..', 'model'))
sys.path.append(model_dir)

from run_enkf_rk4 import run_EnKF_rk4


# アンサンブル数Nごとの比較実験
def main():
    try:
        eta0 = np.loadtxt("../../../data/wave_height_initial.csv",
                          delimiter=",", skiprows=1, usecols=1)
        H_minus = np.loadtxt("../../../data/bathemetry.csv",
                             delimiter=",", skiprows=1, usecols=1)

        Ns = [2, 10, 30, 50, 75, 100]
        max_heights = []
        arrival_times = []
        true_max_h = None
        true_arrival_t = None

        for N in Ns:
            print(f"N={N}...")
            stats_true, stats_ana = run_EnKF_rk4(eta0, H_minus,
                                                 obs_space_interval=60,
                                                 obs_time_interval_sec=3.0,
                                                 N=N,
                                                 inflation_factor=1.05,
                                                 save_dir="../../../result_rk4/N_compare"
                                                 )
            max_heights.append(stats_ana['max_height'])
            arrival_times.append(stats_ana['arrival_time'])

            # 真値はどのNでも共通なので一度だけ保持
            if true_max_h is None:
                true_max_h = stats_true['max_height']
                true_arrival_t = stats_true['arrival_time']

        # --- (a) 津波到達時刻のプロット ---
        plt.figure(figsize=(7, 5))
        # 収束圏の網掛け (±20s)
        plt.fill_between([0, 105], true_arrival_t - 20, true_arrival_t + 20,
                         color='gray', alpha=0.1,
                         label='Convergence Zone (±20s)'
                         )

        # 推定値と真値
        plt.plot(Ns, arrival_times, marker='o', color='#1f77b4',
                 label='Estimation', zorder=3
                 )
        plt.axhline(y=true_arrival_t, color='r', linestyle='--',
                    label=f'Numerical Truth ({true_arrival_t:.2f}s)')

        plt.xlabel('Ensemble Size (N)')
        plt.ylabel('Arrival Time (s)')
        plt.xlim(0, 105)
        plt.ylim(0, 1200)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc='upper right', fontsize='small')

        plt.savefig("../../../result_rk4/N_compare/convergence_arrival_time.png")
        plt.close()
        print("Saved: convergence_arrival_time.png")

        # --- (b) 最大波高のプロット ---
        plt.figure(figsize=(7, 5))

        # 収束圏の網掛け (±0.2m)
        plt.fill_between([0, 105], true_max_h - 0.2, true_max_h + 0.2,
                         color='gray', alpha=0.1,
                         label='Convergence Zone (±0.2m)'
                         )

        # 推定値と真値
        plt.plot(Ns, max_heights, marker='o', color='#1f77b4',
                 label='Estimation', zorder=3
                 )
        plt.axhline(y=true_max_h, color='r', linestyle='--',
                    label=f'Numerical Truth ({true_max_h:.2f}m)')

        plt.xlabel('Ensemble Size (N)')
        plt.ylabel('Max Height (m)')
        plt.xlim(0, 105)
        plt.ylim(0, 4.0)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc='upper right', fontsize='small')

        plt.tight_layout()
        plt.savefig("../../../result_rk4/N_compare/convergence_max_height.png")
        print("\nSuccess: Sensitivity graphs saved.")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
