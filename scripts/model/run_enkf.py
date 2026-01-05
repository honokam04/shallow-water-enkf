import numpy as np
import matplotlib.pyplot as plt
import os


from shallow_water import step_forward
from prepare import pack, unpack, create_H_matrix
from EnKF import analysis_step


def run_EnKF(eta_raw, H_raw, g, dx, total_time, CFL, save_dir="../../results"):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    eta = eta_raw.copy()             # 初期水位 η(x,0)
    H = np.abs(H_raw) * 1000.0     # 水深 [m]
    nx = len(H)      # 空間格子点数

    u = np.zeros(nx + 1)    # スタガード格子流速

    eta_new, u_new, dt = step_forward(eta, u, H, nx, g, dx, CFL)
    nt = int(total_time / dt)

    # 真の履歴を保存
    eta_true_history = np.zeros((nt, nx))
    u_true_history = np.zeros((nt, nx + 1))

    eta_curr = eta_raw.copy()
    u_curr = np.zeros(nx + 1)

    for t_step in range(nt):
        eta_curr, u_curr, _ = step_forward(eta_curr, u_curr, H, nx, g, dx, CFL)

        # 履歴保存
        eta_true_history[t_step, :] = eta_curr
        u_true_history[t_step, :] = u_curr

    # 結果の保存
    np.save(os.path.join(save_dir, "true_history.npy"), eta_true_history)

    # 観測行列 H
    H_mat, obs_indices, n_state, n_obs = create_H_matrix(
        nx, interval=40, missing_idx=None
        )   # 40 × 0.5km = 20 km 間隔

    # 観測ノイズの共分散 R (5cm程度の誤差を想定)
    sigma_obs = 0.05
    R = np.eye(len(obs_indices)) * (sigma_obs**2)

    # 全時刻分の観測データ (nt, n_obs)
    y_obs_history = np.zeros((nt, len(obs_indices)))

    for t in range(nt):
        # 状態ベクトルの生成
        v_true = pack(eta_true_history[t, :], u_true_history[t, :])

        # 観測ベクトルの生成
        y_pure = H_mat @ v_true
        noise = np.random.normal(0, sigma_obs, len(obs_indices))
        y_obs_history[t, :] = y_pure + noise

    print(f"Observation data generated: {y_obs_history.shape}")

    # データ同化結果保存用
    eta_analysis_history = np.zeros((nt, nx))

    N = 40   # アンサンブル数

    # 初期アンサンブルの用意
    U = np.zeros((n_state, N))
    v_initial = pack(eta_raw, np.zeros(nx+1))

    for m in range(N):
        # 初期アンサンブルとして真値にランダムな摂動を与える
        U[:, m] = v_initial + np.random.normal(0, 0.15, n_state)

    # 観測を入れる時間間隔を指定
    obs_interval_sec = 3.0  # 観測を入れる間隔 (秒)
    obs_step_interval = int(obs_interval_sec / dt)  # ステップ数に換算

    # メインループ
    for t in range(nt):
        # 観測の情報を同化
        if t % obs_step_interval == 0:
            y_now = y_obs_history[t]
            U = analysis_step(U, y_now, n_obs, H_mat, R, N)
        else:
            pass    # 観測がないときは、修正せずそのまま推定値を用いる

        # 結果の保存
        m_analysis = np.mean(U, axis=1)  # 推定値
        eta_ana, _ = unpack(m_analysis, nx)
        eta_analysis_history[t, :] = eta_ana  # etaをヒートマップ用に保存

        # 1ステップ先へ進める
        for m in range(N):
            eta_m, u_m = unpack(U[:, m], nx)
            eta_m, u_m, _ = step_forward(eta_m, u_m, H, nx, g, dx, CFL)

            # 進めた結果をアンサンブルへ
            U[:, m] = pack(eta_m, u_m)

    # 結果の保存
    np.save(os.path.join(save_dir, "enkf_history.npy"), eta_analysis_history)

    print("Data Assimilation completed safely")

    x_axis = np.linspace(0, 250, nx)
    t_axis = np.linspace(0, nt * dt, nt)
    X_mesh, T_mesh = np.meshgrid(x_axis, t_axis)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)

    vmin, vmax = -3.0, 3.0

    # 1. 真値 (True State)
    im1 = axes[0].pcolormesh(X_mesh, T_mesh, eta_true_history, cmap='jet',
                             vmin=vmin, vmax=vmax, shading='auto')
    axes[0].set_title('True State (Reference)')
    axes[0].set_ylabel('Time (s)')
    axes[0].set_xlabel('Distance from Coast (km)')

    # 2. 同化結果 (EnKF Analysis)
    im2 = axes[1].pcolormesh(X_mesh, T_mesh, eta_analysis_history, cmap='jet',
                            vmin=vmin, vmax=vmax, shading='auto')
    axes[1].set_title('Data Assimilation (EnKF, 30km interval)')
    axes[1].set_xlabel('Distance from Coast (km)')

    # 観測地点を縦線（垂直線）で表示
    for idx in obs_indices:
        axes[1].axvline(x=idx * dx / 1000, color='white',
                        alpha=0.3, linestyle='--', linewidth=0.8)

    fig.colorbar(im1, ax=axes.ravel().tolist(),
                 label='Wave Height (m)', shrink=0.8)

    plt.savefig(os.path.join(save_dir, "compare_heatmap.png"))
    print(f"Results saved to {save_dir}")


def main():
    eta0_path = "../../data/wave_height_initial.csv"
    H_path = "../../data/bathemetry.csv"

    try:
        eta0 = np.loadtxt(eta0_path, delimiter=",", skiprows=1, usecols=1)
        H_minus = np.loadtxt(H_path, delimiter=",", skiprows=1, usecols=1)

        run_EnKF(eta0, H_minus, g=9.81, dx=500.0, total_time=3000.0, CFL=0.5)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
