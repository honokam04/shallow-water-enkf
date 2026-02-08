import numpy as np
import matplotlib.pyplot as plt
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.abspath(os.path.join(current_dir, '..', 'model'))
sys.path.append(model_dir)

from run_enkf import run_EnKF


# inflation_factorごとの比較実験
def main():
    try:
        eta0 = np.loadtxt("../../../data/wave_height_initial.csv",
                          delimiter=",", skiprows=1, usecols=1)
        H_minus = np.loadtxt("../../../data/bathemetry.csv",
                             delimiter=",", skiprows=1, usecols=1)

        infs = [1.1, 1.05, 1.02]
        for inf in infs:
            print(f"inflation_rate={inf}...")
            run_EnKF(eta0, H_minus, obs_space_interval=60,
                     obs_time_interval_sec=3.0, N=10, inflation_factor=inf,
                     save_dir="../../../result/inf_compare")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
