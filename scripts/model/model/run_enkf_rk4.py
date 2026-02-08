import numpy as np
import matplotlib.pyplot as plt
import os

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from shallow_water import step_forward_rk4
from prepare import pack, unpack, create_H_matrix, get_true_state_rk4
from prepare import create_observations, init_ensemble, visualize_results
from analysis import analyze_and_compare_tsunami
from EnKF import analysis_step


def run_EnKF_rk4(eta_raw, H_raw, obs_space_interval,
                 obs_time_interval_sec,
                 N, g=9.81, dx=500.0, total_time=3000.0,
                 inflation_factor=1.05,
                 CFL=0.01, save_dir="../../../result_rk4/EnKF"):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.random.seed(42)  # シード固定

    # 0. 基本パラメータの計算
    H = np.abs(H_raw) * 1000.0  # 水深 H を m 単位に
    nx = len(H)  # 格子点数
    dt = CFL * dx / np.sqrt(g * np.max(H))  # CFL条件から時間刻みを計算
    print(f"dt:{dt}(s)")
    nt = int(total_time / dt)  # 時間刻み数

    # 1. 真値データの生成
    eta_true, u_true = get_true_state_rk4(eta_raw, H, nx, nt, g, dx, CFL)
    np.save(os.path.join(save_dir, "true_history.npy"), eta_true)

    # 2. 観測データの準備
    H_mat, obs_indices, _, n_obs = create_H_matrix(
        nx, interval=obs_space_interval
        )
    gamma = 0.05  # 観測ノイズの標準偏差
    Gamma = np.eye(n_obs) * (gamma**2)  # 観測ノイズの共分散
    y_obs_history = create_observations(eta_true, u_true, H_mat,
                                        gamma, n_obs, nt)  # 観測 y を生成

    # 3. EnKFの実行
    N = N  # アンサンブル数
    U = init_ensemble(nx, dx, N)  # 初期アンサンブル
    obs_step_interval = int(obs_time_interval_sec / dt)  # 観測1回の間に何ステップ進むか
    eta_analysis_history = np.zeros((nt, nx))  # データ同化結果を記録

    for t in range(nt):
        # 1. 予測ステップ (Prediction)
        for m in range(N):
            # 時間発展
            eta_m, u_m = unpack(U[:, m], nx)
            eta_m, u_m, _ = step_forward_rk4(eta_m, u_m, H, nx, g, dx, CFL)
            U[:, m] = pack(eta_m, u_m)  # アンサンブルごとの推定値

        # 2. 分析ステップ (Analysis) - 観測があるタイミングのみ
        if t % obs_step_interval == 0:
            # inflationを適用する場合はコメントアウトを外す
            U_mean = np.mean(U, axis=1, keepdims=True)
            U = U_mean + np.sqrt(inflation_factor) * (U - U_mean)
            # 推定値 U を観測 y で更新
            U = analysis_step(U, y_obs_history[t], n_obs, H_mat, Gamma, N)

        # 3. 推定値の保存 (記録するのは観測を取り込んだ後の最新の状態)
        m_analysis = np.mean(U, axis=1)  # 推定値
        eta_ana, _ = unpack(m_analysis, nx)
        eta_analysis_history[t, :] = eta_ana  # 記録

    # 4. 結果の保存と可視化
    np.save(os.path.join(
        save_dir,
        # f"enkf_history_dx={obs_space_interval / 2}_CFL={CFL}_N={N}.npy"
        # inflationを行う場合
        f"enkf_history_dx={obs_space_interval / 2}_CFL={CFL}_N={N}_inf={inflation_factor}.npy"
        ),
        eta_analysis_history
        )
    visualize_results(eta_true, eta_analysis_history,
                      obs_indices, dx, dt, nt, nx, N,
                      os.path.join(
                          save_dir,
                          # f"dx={obs_space_interval / 2}_CFL={CFL}_N={N}.png"
                          # inflationを行う場合
                          f"dx={obs_space_interval / 2}_CFL={CFL}_N={N}_inf={inflation_factor}.png"
                          ))

    print(f"Success: Results saved to {save_dir}")

    # 結果の出力
    return analyze_and_compare_tsunami(
        eta_true, eta_analysis_history, dt
        )


def main():
    try:
        eta0 = np.loadtxt("../../../data/wave_height_initial.csv",
                          delimiter=",", skiprows=1, usecols=1)
        H_minus = np.loadtxt("../../../data/bathemetry.csv",
                             delimiter=",", skiprows=1, usecols=1)
        run_EnKF_rk4(eta0, H_minus, obs_space_interval=60,
                     # 観測点間隔はobs_space_interval × 0.5 (km)
                     obs_time_interval_sec=3.0, N=100, inflation_factor=1.05,
                     CFL=0.01, save_dir="../../../result_rk4/EnKF_inf")
        # inflationを行う場合はsave_dir="../../../result/EnKF_inf"
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
