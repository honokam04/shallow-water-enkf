import numpy as np
import matplotlib.pyplot as plt
import os

from shallow_water import step_forward_rk4
from prepare import pack, unpack, create_H_matrix, get_true_state_rk4
from prepare import create_observations, init_ensemble, visualize_results
from EnKF import analysis_step


def run_EnKF_real(eta_raw, H_raw, obs_space_interval,
                  obs_time_interval_sec,
                  N, g=9.81, dx=500.0, total_time=3000.0,
                  assimilation_end_time=750.0,  # 同化を終了する境界時間
                  inflation_factor=1.05,
                  CFL=0.01, save_dir="../../../result/EnKF"):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.random.seed(42)

    # 0. 基本パラメータの計算
    H = np.abs(H_raw) * 1000.0
    nx = len(H)
    dt = CFL * dx / np.sqrt(g * np.max(H))
    print(f"dt:{dt}(s)")
    nt = int(total_time / dt)

    # 同化を停止するステップ数を計算
    assimilation_end_step = int(assimilation_end_time / dt)

    # 1. 真値データの生成
    eta_true, u_true = get_true_state_rk4(eta_raw, H, nx, nt, g, dx, CFL)
    np.save(os.path.join(save_dir, "true_history.npy"), eta_true)

    # 2. 観測データの準備
    H_mat, obs_indices, _, n_obs = create_H_matrix(
        nx, interval=obs_space_interval
        )
    gamma = 0.05
    Gamma = np.eye(n_obs) * (gamma**2)
    y_obs_history = create_observations(
        eta_true, u_true, H_mat, gamma, n_obs, nt
        )

    # 3. EnKFの実行
    U = init_ensemble(nx, dx, N)
    obs_step_interval = int(obs_time_interval_sec / dt)
    eta_analysis_history = np.zeros((nt, nx))

    for t in range(nt):
        # --- 予測ステップ (Prediction) ---
        # どの時間帯でもアンサンブルメンバー全員を1ステップ進める
        for m in range(N):
            eta_m, u_m = unpack(U[:, m], nx)
            eta_m, u_m, _ = step_forward_rk4(eta_m, u_m, H, nx, g, dx, CFL)
            U[:, m] = pack(eta_m, u_m)

        # --- 分析ステップ (Analysis) ---
        # assimilation_end_time (750秒) 以前、かつ観測タイミングの場合のみ更新を行う
        if t <= assimilation_end_step:
            if t % obs_step_interval == 0:
                # インフレーションの適用
                U_mean = np.mean(U, axis=1, keepdims=True)
                U = U_mean + np.sqrt(inflation_factor) * (U - U_mean)
                # 観測データを用いて修正
                U = analysis_step(U, y_obs_history[t], n_obs, H_mat, Gamma, N)
        else:
            # 750秒以降は分析ステップをスキップし、純粋な数値シミュレーション(予測)に移行
            pass

        # 推定値（アンサンブル平均）の記録
        m_analysis = np.mean(U, axis=1)
        eta_ana, _ = unpack(m_analysis, nx)
        eta_analysis_history[t, :] = eta_ana

    # 4. 結果の保存と可視化
    file_name = f"enkf_pred_N={N}_end={assimilation_end_time}.npy"
    np.save(os.path.join(save_dir, file_name), eta_analysis_history)

    visualize_results(eta_true, eta_analysis_history,
                      obs_indices, dx, dt, nt, nx, N,
                      os.path.join(save_dir, file_name.replace(".npy", ".png")))

    print(f"Success: Results saved to {save_dir}")

    eta_at_coast = eta_analysis_history[:, 0]
    eta_true_at_coast = eta_true[:, 0]

    return eta_at_coast, eta_true_at_coast, dt


def main():
    try:
        eta0 = np.loadtxt("../../../data/wave_height_initial.csv",
                          delimiter=",", skiprows=1, usecols=1)
        H_minus = np.loadtxt("../../../data/bathemetry.csv",
                             delimiter=",", skiprows=1, usecols=1)

        run_EnKF_real(eta0, H_minus,
                      obs_space_interval=60,
                      obs_time_interval_sec=3.0,
                      N=50,
                      total_time=3000.0,
                      assimilation_end_time=750.0,  # ここで同化終了時間を指定
                      inflation_factor=1.05,
                      CFL=0.01,
                      save_dir="../../../result/rk4/EnKF_pred")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
