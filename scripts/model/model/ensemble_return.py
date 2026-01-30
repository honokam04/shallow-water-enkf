import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

from prepare import get_true_state, create_H_matrix, create_observations
from prepare import init_ensemble, unpack, pack
from shallow_water import step_forward
from EnKF import analysis_step
from analysis import analyze_and_compare_tsunami


# アンサンブルごとの履歴を返却
def run_EnKF_with_ensemble_history(eta_raw, H_raw, obs_space_interval,
                                   obs_time_interval_sec,
                                   N, g=9.81, dx=500.0, total_time=3000.0,
                                   inflation_factor=1.05,
                                   CFL=0.01):
    # 0. 基本パラメータの計算
    H = np.abs(H_raw) * 1000.0
    nx = len(H)
    dt = CFL * dx / np.sqrt(g * np.max(H))
    nt = int(total_time / dt)

    # 1. 真値と観測の準備
    eta_true, u_true = get_true_state(eta_raw, H, nx, nt, g, dx, CFL)
    H_mat, _, _, n_obs = create_H_matrix(nx, interval=obs_space_interval)
    gamma = 0.05
    Gamma = np.eye(n_obs) * (gamma**2)
    y_obs_history = create_observations(
        eta_true, u_true, H_mat, gamma, n_obs, nt
        )

    # 2. EnKFの初期化
    U = init_ensemble(nx, dx, N)
    obs_step_interval = int(obs_time_interval_sec / dt)

    # 各アンサンブルの水位履歴を保存する配列 ---
    # Shape: (時間ステップ, 格子点数, アンサンブルメンバー数)
    ensemble_history = np.zeros((nt, nx, N))
    # ----------------------------------------------

    for t in range(nt):
        # 1. 予測ステップ
        for m in range(N):
            eta_m, u_m = unpack(U[:, m], nx)
            eta_m, u_m, _ = step_forward(eta_m, u_m, H, nx, g, dx, CFL)
            U[:, m] = pack(eta_m, u_m)

        # 2. 分析ステップ
        if t % obs_step_interval == 0:
            U_mean = np.mean(U, axis=1, keepdims=True)
            U = U_mean + np.sqrt(inflation_factor) * (U - U_mean)
            U = analysis_step(U, y_obs_history[t], n_obs, H_mat, Gamma, N)

        # --- 追加: 現時刻の全アンサンブルの eta を抽出して保存 ---
        for m in range(N):
            eta_m_ana, _ = unpack(U[:, m], nx)
            ensemble_history[t, :, m] = eta_m_ana
        # ----------------------------------------------------

    # 3. 既存の統計処理
    m_analysis = np.mean(U, axis=1)
    eta_analysis_history = np.mean(ensemble_history, axis=2)  # 全メンバーの平均

    stats_true, stats_ana = analyze_and_compare_tsunami(
        eta_true, eta_analysis_history, dt
    )

    return stats_true, stats_ana, ensemble_history


# 真値とアンサンブル平均のみのアニメーション
def save_ensemble_animation(
        ensemble_history, eta_true, dx, dt, save_path, interval=50
        ):
    nt, nx, N = ensemble_history.shape
    x_axis = np.arange(nx) * dx / 1000.0  # km単位

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.set_xlim(0, x_axis[-1])
    ymin, ymax = np.min(eta_true) * 1.2, np.max(eta_true) * 1.2
    ax.set_ylim(ymin, ymax)

    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Wave Height (m)")
    ax.grid(True, alpha=0.3)

    # 描画オブジェクトのリスト
    ens_lines = [ax.plot([], [], color='gray', alpha=0.2, lw=0.5)[0] for _ in range(N)]
    # アンサンブル平均（赤）
    mean_line, = ax.plot([], [], color='red', lw=2, label='Ensemble Mean')
    # 真値（黒の破線）
    true_line, = ax.plot([], [], color='black', ls='--', lw=1.5, label='True')

    ax.legend(loc='upper right')
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    def init():
        for line in ens_lines:
            line.set_data([], [])
        mean_line.set_data([], [])
        true_line.set_data([], [])
        time_text.set_text('')
        return ens_lines + [mean_line, true_line, time_text]

    def update(t):
        # 各メンバーの更新
        for m in range(N):
            ens_lines[m].set_data(x_axis, ensemble_history[t, :, m])

        # 平均値の計算と更新
        mean_eta = np.mean(ensemble_history[t, :, :], axis=1)
        mean_line.set_data(x_axis, mean_eta)

        # 真値の更新
        true_line.set_data(x_axis, eta_true[t, :])

        time_text.set_text(f'Time: {t * dt:.1f} s')
        return ens_lines + [mean_line, true_line, time_text]

    # アニメーションの作成
    ani = animation.FuncAnimation(
        fig, update, frames=range(0, nt, max(1, nt//200)),
        init_func=init, blit=True
    )

    # 保存処理
    print(f"Saving animation to {save_path}...")
    if save_path.endswith('.gif'):
        ani.save(save_path, writer='pillow', fps=20)
    else:
        ani.save(save_path, writer='ffmpeg', fps=20)

    plt.close()
    print("Success: Animation saved.")


def save_all_members_animation(
        ensemble_history, eta_true, obs_indices, dx, dt, save_path
        ):

    nt, nx, N = ensemble_history.shape
    x_axis = np.arange(nx) * dx / 1000.0  # 距離 (km)

    fig, ax = plt.subplots(figsize=(10, 6))

    # --- 1. スタイルの設定 ---
    ax.set_xlim(0, x_axis[-1])
    padding = 1.5
    ax.set_ylim(np.min(eta_true) * padding, np.max(eta_true) * padding)
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Wave Height (m)")
    ax.grid(True, alpha=0.3)

    # --- 2. 観測地点の描画 ---
    for idx in obs_indices:
        ax.axvline(x=idx * dx / 1000.0, color='blue', alpha=0.15, ls=':',
                   lw=1, label='Obs Point' if idx == obs_indices[0] else ""
                   )

    # --- 3. プロットオブジェクトの作成 ---
    if N == 2:
        colors = ['#00FFFF', '#FF00FF']
        lws = [2.0, 1.0]
        alphas = [0.6, 0.9]
    else:
        # Nが多い時はカラーマップを使用
        cmap = plt.get_cmap('viridis')
        colors = [cmap(i/N) for i in range(N)]
        lws = [0.8] * N
        alphas = [0.5] * N

    ens_lines = []
    for i in range(N):
        line, = ax.plot(
            [], [], color=colors[i], alpha=alphas[i], lw=lws[i],
            label=f'Member {i+1}'
            )
        ens_lines.append(line)

    # 真値（黒の太い破線）
    true_line, = ax.plot([], [], color='black', ls='--', lw=2,
                         zorder=10, label='True (Target)'
                         )

    ax.legend(loc='upper right', fontsize='small', ncol=2 if N > 5 else 1)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, weight='bold')

    # --- 4. アニメーション関数 ---
    def init():
        for line in ens_lines:
            line.set_data([], [])
        true_line.set_data([], [])
        time_text.set_text('')
        return ens_lines + [true_line, time_text]

    def update(t):
        # 各メンバーの波形を更新
        for m in range(N):
            ens_lines[m].set_data(x_axis, ensemble_history[t, :, m])

        # 真値の波形を更新
        true_line.set_data(x_axis, eta_true[t, :])

        # 時刻表示の更新
        time_text.set_text(f'Time: {t * dt:.1f} s')
        return ens_lines + [true_line, time_text]

    # --- 5. 保存 ---
    step = max(1, nt // 200)
    ani = animation.FuncAnimation(
        fig, update, frames=range(0, nt, step),
        init_func=init, blit=True
    )

    print(f"Starting to save animation to: {save_path}")
    if save_path.endswith('.gif'):
        ani.save(save_path, writer='pillow', fps=15)
    else:
        ani.save(save_path, writer='ffmpeg', fps=15)

    plt.close()
    print("Animation save complete.")


def main():
    save_dir = "../../../result/EnKF_animation"
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    np.random.seed(42)

    try:
        # データ読み込み
        eta0 = np.loadtxt("../../../data/wave_height_initial.csv",
                          delimiter=",", skiprows=1, usecols=1
                          )
        H_minus = np.loadtxt("../../../data/bathemetry.csv",
                             delimiter=",", skiprows=1, usecols=1
                             )

        # 1. EnKF実行
        N_members = 2
        obs_interval = 60
        stats_true, stats_ana, ens_hist = run_EnKF_with_ensemble_history(
            eta0, H_minus, obs_space_interval=obs_interval,
            obs_time_interval_sec=3.0, N=N_members
        )

        # 2. パラメータ特定と観測行列の再取得
        dx, CFL, g = 500.0, 0.01, 9.81
        H = np.abs(H_minus) * 1000.0
        dt = CFL * dx / np.sqrt(g * np.max(H))
        _, obs_indices, _, _ = create_H_matrix(len(H), interval=obs_interval)

        # 真値の生成
        eta_true, _ = get_true_state(
            eta0, H, len(H), ens_hist.shape[0], g, dx, CFL
            )

        # 3. アニメーション保存
        save_all_members_animation(
            ens_hist, eta_true, obs_indices, dx, dt,
            os.path.join(save_dir, f"ensemble_bundle_N={N_members}.gif")
        )

        print("Done!")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
