import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from shallow_water import step_forward
from prepare import pack, unpack, create_H_matrix
from EnKF import analysis_step


# --- 1. 指標判定関数 (到達時刻 & 最大波高) ---
def get_metrics(eta_history, dt, threshold=0.2):
    coastal_series = eta_history[:, 0]
    # 到達時刻
    arrival_indices = np.where(coastal_series >= threshold)[0]
    arrival_time = float(arrival_indices[0] * dt) if len(arrival_indices) > 0 else None
    # 最大波高
    max_height = float(np.max(coastal_series))
    return arrival_time, max_height


# --- 2. 真値（正解）の生成 ---
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


# --- 3. 観測データの生成 ---
def create_observations(eta_true, u_true, H_mat, gamma, n_obs, nt):
    y_obs_history = np.zeros((nt, n_obs))
    for t in range(nt):
        v_true = pack(eta_true[t, :], u_true[t, :])
        noise = np.random.normal(0, gamma, n_obs)
        y_obs_history[t, :] = (H_mat @ v_true) + noise
    return y_obs_history


# --- 4. 初期アンサンブルの作成 ---
def init_ensemble(nx, dx, N, sigma_init=0.15, correlation_length=20000.0):
    n_state = 2 * nx + 1
    U = np.zeros((n_state, N))
    x = np.arange(nx) * dx
    X1, X2 = np.meshgrid(x, x)
    dist_matrix = np.abs(X1 - X2)
    B = np.exp(-(dist_matrix**2) / (2 * correlation_length**2))
    mean = np.zeros(nx)
    C0 = B * (sigma_init**2)
    for m in range(N):
        eta_init_member = np.random.multivariate_normal(mean, C0)
        u_init_member = np.zeros(nx + 1)
        U[:, m] = pack(eta_init_member, u_init_member)
    return U


# --- 5. 結果の可視化 ---
def visualize_results(eta_true, eta_analysis, obs_indices,
                      dx, dt, nt, nx, N, save_path
                      ):
    x_axis = np.linspace(0, (nx*dx)/1000, nx)
    t_axis = np.linspace(0, nt * dt, nt)
    X_mesh, T_mesh = np.meshgrid(x_axis, t_axis)

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    im = ax.pcolormesh(X_mesh, T_mesh, eta_analysis, cmap='jet',
                       vmin=-3.0, vmax=3.0, shading='auto'
                       )
    ax.set_title(f'EnKF with Inflation (N={N})')
    ax.set_ylabel('Time (s)'); ax.set_xlabel('Distance (km)')
    for idx in obs_indices:
        ax.axvline(x=idx * dx / 1000, color='white', alpha=0.3,
                   linestyle='--', linewidth=0.8
                   )
    fig.colorbar(im, ax=ax, label='Wave Height (m)', shrink=0.8)
    plt.savefig(save_path); plt.close()


# --- メイン実行関数 ---
def run_EnKF_with_inflation(eta_raw, H_raw, obs_space_interval,
                            obs_time_interval_sec, N, inflation_factor=1.1,
                            g=9.81, dx=500.0, total_time=3000.0,
                            CFL=0.01, save_dir="../../results"
                            ):

    if not os.path.exists(save_dir): os.makedirs(save_dir)
    np.random.seed(42)  # シード固定

    # 0. パラメータ準備
    H = np.abs(H_raw) * 1000.0
    nx = len(H)
    dt = CFL * dx / np.sqrt(g * np.max(H))
    nt = int(total_time / dt)

    # 1. 真値データの生成
    print(f"Generating Truth (CFL={CFL})...")
    eta_true, u_true = get_true_state(eta_raw, H, nx, nt, g, dx, CFL)
    true_arrival, true_max_h = get_metrics(eta_true, dt)

    # 2. 観測データの準備
    H_mat, obs_indices, _, n_obs = create_H_matrix(nx,
                                                   interval=obs_space_interval
                                                   )
    gamma = 0.05
    Gamma = np.eye(n_obs) * (gamma**2)
    y_obs_history = create_observations(eta_true, u_true,
                                        H_mat, gamma, n_obs, nt
                                        )

    # 3. EnKF実行 (共分散膨張あり)
    print(f"Running EnKF (N={N}, Inflation={inflation_factor})...")
    U = init_ensemble(nx, dx, N)
    obs_step_interval = int(obs_time_interval_sec / dt)
    eta_analysis_history = np.zeros((nt, nx))
    rmse_sum = 0

    for t in range(nt):
        # 予測ステップ
        for m in range(N):
            eta_m, u_m = unpack(U[:, m], nx)
            eta_m, u_m, _ = step_forward(eta_m, u_m, H, nx, g, dx, CFL)
            U[:, m] = pack(eta_m, u_m)

        # 分析ステップ
        if t % obs_step_interval == 0:
            # ★ 共分散膨張の適用
            U_mean = np.mean(U, axis=1, keepdims=True)
            U = U_mean + np.sqrt(inflation_factor) * (U - U_mean)

            # 通常の分析
            U = analysis_step(U, y_obs_history[t], n_obs, H_mat, Gamma, N)

        # 記録
        m_analysis = np.mean(U, axis=1)
        eta_ana, _ = unpack(m_analysis, nx)
        eta_analysis_history[t, :] = eta_ana
        rmse_sum += np.sqrt(np.mean((eta_ana - eta_true[t, :])**2))

    # 指標の計算
    est_arrival, est_max_h = get_metrics(eta_analysis_history, dt)
    avg_rmse = rmse_sum / nt

    # CSV保存
    csv_path = os.path.join(save_dir,
                            f"inflation_test_N={N}_inf={inflation_factor}.csv"
                            )
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["N", "Inflation", "True_Arrival", "Est_Arrival",
                         "True_MaxH", "Est_MaxH", "Avg_RMSE"])
        writer.writerow([N, inflation_factor, true_arrival, est_arrival,
                         true_max_h, est_max_h, avg_rmse])

    # 可視化
    visualize_results(eta_true, eta_analysis_history, obs_indices,
                      dx, dt, nt, nx, N,
                      os.path.join(save_dir,
                                   f"heatmap_N={N}_inf={inflation_factor}.png"
                                   ))

    print(f"Metrics: RMSE={avg_rmse:.4f}, MaxH Error={est_max_h - true_max_h:.4f}")
    return avg_rmse, est_max_h


def main():
    try:
        eta0 = np.loadtxt("../../data/wave_height_initial.csv",
                          delimiter=",", skiprows=1, usecols=1
                          )
        H_raw = np.loadtxt("../../data/bathemetry.csv",
                           delimiter=",", skiprows=1, usecols=1
                           )

        # Inflation 1.05 (5%増)
        run_EnKF_with_inflation(eta0, H_raw, obs_space_interval=60,
                                obs_time_interval_sec=3.0, N=50,
                                inflation_factor=1.05
                                )

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
