import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from shallow_water import step_forward, step_forward_rk4


# etaとuを状態ベクトルとしてまとめる
def pack(eta, u):
    return np.concatenate([eta, u])


# 状態ベクトルをetaとuに分解する
def unpack(v, nx):
    # v の前半 nx 個が eta, 後半 nx+1 個が u
    eta = v[:nx]
    u = v[nx:]
    return eta, u


# 観測行列の作成
def create_H_matrix(nx, interval, missing_idx=None):
    n_state = 2 * nx + 1    # 状態次元
    # 1. 観測候補点のインデックス
    obs_indices = np.arange(0, nx, interval)

    # 2. 欠測点を除いた観測点インデックスのリストを生成
    if missing_idx is None:
        missing_idx = []
    obs_indices = [idx for idx in obs_indices if idx not in missing_idx]

    n_obs = len(obs_indices)  # 実際の観測次元

    # 3. 観測行列の生成（観測した点に対応する成分を1に）
    H = np.zeros((n_obs, n_state))
    for i, idx in enumerate(obs_indices):
        H[i, idx] = 1

    return H, np.array(obs_indices), n_state, n_obs


# 真値の作成
def get_true_state(eta_raw, H, nx, nt, g, dx, CFL):
    # 配列の用意
    eta_true = np.zeros((nt, nx))
    u_true = np.zeros((nt, nx + 1))

    # 真値の初期条件
    e_curr = eta_raw.copy()  # η(x,0)
    u_curr = np.zeros(nx + 1)  # u(x,0)=0 を指定（まだ動いていないと仮定）

    # 1D浅水波モデルによる数値計算
    for t in range(nt):
        e_curr, u_curr, _ = step_forward(e_curr, u_curr, H, nx, g, dx, CFL)
        # 現在の値を記録
        eta_true[t, :] = e_curr
        u_true[t, :] = u_curr

    return eta_true, u_true


# 真値の作成
def get_true_state_rk4(eta_raw, H, nx, nt, g, dx, CFL):
    # 配列の用意
    eta_true = np.zeros((nt, nx))
    u_true = np.zeros((nt, nx + 1))

    # 真値の初期条件
    e_curr = eta_raw.copy()  # η(x,0)
    u_curr = np.zeros(nx + 1)  # u(x,0)=0 を指定（まだ動いていないと仮定）

    # 1D浅水波モデルによる数値計算
    for t in range(nt):
        e_curr, u_curr, _ = step_forward_rk4(e_curr, u_curr, H, nx, g, dx, CFL)
        # 現在の値を記録
        eta_true[t, :] = e_curr
        u_true[t, :] = u_curr

    return eta_true, u_true


# 観測データの生成
def create_observations(eta_true, u_true, H_mat, gamma, n_obs, nt):
    y_obs_history = np.zeros((nt, n_obs))

    for t in range(nt):
        v_true = pack(eta_true[t, :], u_true[t, :])  # 状態ベクトルを生成
        noise = np.random.normal(0, gamma, n_obs)  # 観測ノイズの生成
        y_obs_history[t, :] = (H_mat @ v_true) + noise  # 観測を表現・記録
    return y_obs_history


# 初期アンサンブルの作成
def init_ensemble(nx, dx, N, sigma_init=0.15, correlation_length=20000.0):
    n_state = 2 * nx + 1  # 状態次元
    U = np.zeros((n_state, N))

    # 1. 距離行列の作成（全ての格子点間の距離を計算）
    x = np.arange(nx) * dx  # 空間座標 (m) の生成
    X1, X2 = np.meshgrid(x, x)
    dist_matrix = np.abs(X1 - X2)

    # 2. ガウス型相関行列の作成 (B)
    # exp(- (距離^2) / (2 * 特徴的長さ^2))：「周辺20km圏内くらいまで水位誤差は近い」
    B = np.exp(-(dist_matrix**2) / (2 * correlation_length**2))

    # 3. 初期誤差共分散行列（C0）の生成
    mean = np.zeros(nx)
    C0 = B * (sigma_init**2)

    for m in range(N):
        # 平均0、共分散 B*sigma^2 の多変量正規分布からノイズをサンプリング
        eta_init_member = np.random.multivariate_normal(mean, C0)
        # 流速 u: 先行研究に従い初期値は 0
        u_init_member = np.zeros(nx + 1)

        U[:, m] = pack(eta_init_member, u_init_member)  # アンサンブルとして格納

    return U


# ヒートマップの出力
def visualize_results(eta_true, eta_analysis,
                      obs_indices, dx, dt, nt, nx, N, save_path):
    x_axis = np.linspace(0, (nx*dx)/1000, nx)
    t_axis = np.linspace(0, nt * dt, nt)
    X_mesh, T_mesh = np.meshgrid(x_axis, t_axis)

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    vmin, vmax = -3.0, 3.0

    # 同化結果のプロット
    im = ax.pcolormesh(X_mesh, T_mesh, eta_analysis, cmap='jet',
                       vmin=vmin, vmax=vmax, shading='auto')
    ax.set_title(f'EnKF Data Assimilation (N={N})')
    ax.set_ylabel('Time (s)')
    ax.set_xlabel('Distance (km)')

    # 観測地点（白破線）
    for idx in obs_indices:
        ax.axvline(x=idx * dx / 1000, color='white',
                   alpha=0.8, ls='--', lw=1.5, zorder=5)

    fig.colorbar(im, ax=ax, label='Wave Height (m)', shrink=0.8)
    plt.savefig(save_path)
    plt.close()

    # RMSEの時系列プロット
    rmse = np.sqrt(np.mean((eta_true - eta_analysis)**2, axis=1))

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(t_axis, rmse, label='RMSE', color='tab:blue')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('RMSE (m)')
    ax.set_title(f'Temporal Evolution of Estimation Error(N={N})')
    ax.set_ylim(bottom=0, top=1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')

    ax_inset = inset_axes(ax, width="30%", height="30%",
                          loc='upper right', borderpad=2)

    ax_inset.semilogy(t_axis, rmse, color='tab:red')

    ax_inset.set_title('Log Scale', fontsize=9)
    ax_inset.tick_params(axis='both', which='major', labelsize=8)
    ax_inset.grid(True, which="both", ls="-", alpha=0.2)

    rmse_path = save_path.replace(".png", "_rmse.png")
    plt.savefig(rmse_path)
    plt.close()
