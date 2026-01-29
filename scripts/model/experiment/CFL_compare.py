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

        CFLs = [0.1, 0.05, 0.03, 0.01]
        for CFL in CFLs:
            print(f"CFL={CFL}...")
            run_EnKF(eta0, H_minus, obs_space_interval=60,
                     obs_time_interval_sec=3.0, N=10, inflation_factor=1.05,
                     CFL=CFL, save_dir="../../../result/CFL_compare")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
