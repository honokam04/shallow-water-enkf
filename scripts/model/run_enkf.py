import numpy as np
import matplotlib.pyplot as plt
import os

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from shallow_water import step_forward
from prepare import pack, unpack, create_H_matrix
from EnKF import analysis_step


# --- 1. 真値（正解）の生成 ---
def get_true_state(eta_raw, H, nx, nt, g, dx, CFL):
    eta_true = np.zeros((nt, nx))
    u_true = np.zeros((nt, nx + 1))

    e_curr = eta_raw.copy()
    u_curr = np.zeros(nx + 1)

    for t in range(nt):
        e_curr, u_curr, _ = step_forward(e_curr, u_curr, H, nx, g, dx, CFL)
        eta_true[t, :] = e_curr
        u_true[t, :] = u_curr

    return eta_true, u_true


# --- 2. 観測データの生成 ---
def create_observations(eta_true, u_true, H_mat, sigma_obs, n_obs, nt):
    y_obs_history = np.zeros((nt, n_obs))
    for t in range(nt):
        v_true = pack(eta_true[t, :], u_true[t, :])
        noise = np.random.normal(0, sigma_obs, n_obs)
        y_obs_history[t, :] = (H_mat @ v_true) + noise
    return y_obs_history


# --- 3. 初期アンサンブルの作成 ---
def init_ensemble(nx, dx, N, sigma_init=0.15, correlation_length=20000.0):
    n_state = 2 * nx + 1
    U = np.zeros((n_state, N))

    # 1. 距離行列の作成（各点間の距離を計算）
    x = np.arange(nx) * dx
    X1, X2 = np.meshgrid(x, x)
    dist_matrix = np.abs(X1 - X2)

    # 2. ガウス型相関行列の作成 (P0_block)
    # exp(- (距離^2) / (2 * 特徴的長さ^2))
    B = np.exp(-(dist_matrix**2) / (2 * correlation_length**2))

    # 3. 相関を持ったノイズの生成
    # 平均0、共分散B*sigma^2 の多変量正規分布からサンプリング
    mean = np.zeros(nx)
    cov = B * (sigma_init**2)

    for m in range(N):
        # 水位 eta: 空間相関を持つノイズ
        eta_init_member = np.random.multivariate_normal(mean, cov)
        # 流速 u: 先行研究に従い初期値は 0
        u_init_member = np.zeros(nx + 1)

        U[:, m] = pack(eta_init_member, u_init_member)

    return U


# --- 4. 結果の可視化 ---
def visualize_results(eta_true, eta_analysis,
                      obs_indices, dx, dt, nt, nx, save_path):
    x_axis = np.linspace(0, (nx*dx)/1000, nx)
    t_axis = np.linspace(0, nt * dt, nt)
    X_mesh, T_mesh = np.meshgrid(x_axis, t_axis)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
    vmin, vmax = -3.0, 3.0

    # 真値のプロット
    im1 = axes[0].pcolormesh(X_mesh, T_mesh, eta_true, cmap='jet',
                             vmin=vmin, vmax=vmax, shading='auto')
    axes[0].set_title('True State (Reference)')
    axes[0].set_ylabel('Time (s)')
    axes[0].set_xlabel('Distance (km)')

    # 同化結果のプロット
    im2 = axes[1].pcolormesh(X_mesh, T_mesh, eta_analysis, cmap='jet',
                             vmin=vmin, vmax=vmax, shading='auto')
    axes[1].set_title('EnKF Data Assimilation')
    axes[1].set_xlabel('Distance (km)')

    # 観測地点（白破線）
    for idx in obs_indices:
        axes[1].axvline(x=idx * dx / 1000, color='white',
                        alpha=0.3, linestyle='--', linewidth=0.8)

    fig.colorbar(im1, ax=axes.ravel().tolist(),
                 label='Wave Height (m)', shrink=0.8)
    plt.savefig(save_path)
    plt.close()
    # RMSEの時系列プロット
    rmse = np.sqrt(np.mean((eta_true - eta_analysis)**2, axis=1))

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(t_axis, rmse, label='RMSE (Linear)', color='tab:blue')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('RMSE (m)')
    ax.set_title('Temporal Evolution of Estimation Error')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')

    ax_inset = inset_axes(ax, width="30%", height="30%",
                          loc='upper right', borderpad=2)

    ax_inset.plot(t_axis, rmse, color='tab:red')
    ax_inset.set_yscale('log')
    ax_inset.set_title('Log Scale', fontsize=9)
    ax_inset.tick_params(axis='both', which='major', labelsize=8)
    ax_inset.grid(True, which="both", ls="-", alpha=0.2)

    rmse_path = save_path.replace(".png", "_rmse.png")
    plt.savefig(rmse_path)
    plt.close()


# --- メイン実行関数 ---
def run_EnKF(eta_raw, H_raw, obs_space_interval,  # 空間の観測間隔
             obs_time_interval_sec,  # 時間の観測間隔
             g=9.81, dx=500.0, total_time=3000.0,
             CFL=0.5, save_dir="../../results"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 0. 基本パラメータの計算
    H = np.abs(H_raw) * 1000.0
    nx = len(H)
    dt = CFL * dx / np.sqrt(g * np.max(H))
    nt = int(total_time / dt)

    # 1. 真値の生成
    eta_true, u_true = get_true_state(eta_raw, H, nx, nt, g, dx, CFL)
    np.save(os.path.join(save_dir, "true_history.npy"), eta_true)

    # 2. 観測データの準備
    H_mat, obs_indices, _, n_obs = create_H_matrix(
        nx, interval=obs_space_interval
        )
    sigma_obs = 0.05
    R = np.eye(n_obs) * (sigma_obs**2)
    y_obs_history = create_observations(eta_true, u_true, H_mat,
                                        sigma_obs, n_obs, nt)

    # 3. EnKFの実行
    N = 100
    U = init_ensemble(nx, dx, N)
    obs_step_interval = int(obs_time_interval_sec / dt)
    eta_analysis_history = np.zeros((nt, nx))

    for t in range(nt):
        # 分析（観測による修正）
        if t % obs_step_interval == 0:
            U = analysis_step(U, y_obs_history[t], n_obs, H_mat, R, N)

        # 推定平均値の保存
        m_analysis = np.mean(U, axis=1)
        eta_ana, _ = unpack(m_analysis, nx)
        eta_analysis_history[t, :] = eta_ana

        # 予測（時間進行）
        for m in range(N):
            e_m, u_m = unpack(U[:, m], nx)
            e_m, u_m, _ = step_forward(e_m, u_m, H, nx, g, dx, CFL)
            U[:, m] = pack(e_m, u_m)

    # 4. 結果の保存と可視化
    np.save(os.path.join(
        save_dir,
        f"enkf_history_dx={obs_space_interval}_dt={obs_time_interval_sec}.npy"
        ),
        eta_analysis_history
        )
    visualize_results(eta_true, eta_analysis_history,
                      obs_indices, dx, dt, nt, nx,
                      os.path.join(
                          save_dir,
                          f"dx={obs_space_interval}_dt={obs_time_interval_sec}.png"
                          ))

    print(f"Success: Results saved to {save_dir}")


def main():
    try:
        eta0 = np.loadtxt("../../data/wave_height_initial.csv",
                          delimiter=",", skiprows=1, usecols=1)
        H_minus = np.loadtxt("../../data/bathemetry.csv",
                             delimiter=",", skiprows=1, usecols=1)
        run_EnKF(eta0, H_minus, obs_space_interval=100,  # 観測点間隔は60 × 0.5 (km)
                 obs_time_interval_sec=3.0)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
