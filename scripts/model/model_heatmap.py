import numpy as np
import matplotlib.pyplot as plt
import os

from shallow_water import step_forward


def run_shallow_water(eta0, H_minus, dx=500.0, g=9.81,
                      total_time=3000.0, CFL=0.8, save_dir="../../results"):

    # ディレクトリ作成
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 水深と初期条件の準備
    eta = eta0.copy()
    H = np.abs(H_minus) * 1000.0  # km -> m
    nx = len(H)
    u = np.zeros(nx + 1)

    # --- 1ステップ実行して、正しい dt を取得する ---
    # ループ前に1回空回しするか、手動で一度計算して nt を確定させる必要があります。
    # ここでは物理的な安定性を優先し、最初のステップで決まる dt を基準にします。
    _, _, dt = step_forward(eta, u, H, nx, g, dx, CFL)
    nt = int(total_time / dt)

    print(f"Simulation Info: dx={dx} m, dt={dt:.3f} s, nt={nt} steps")

    # 配列初期化
    eta_history = np.zeros((nt, nx))
    u_history = np.zeros((nt, nx+1))

    # --- 数値計算ループ ---
    current_time = 0.0
    for t_step in range(nt):
        # 戻り値の3つ（eta, u, dt）をすべて受け取る
        eta, u, step_dt = step_forward(eta, u, H, nx, g, dx, CFL)

        # 履歴保存
        eta_history[t_step, :] = eta
        u_history[t_step, :] = u

        current_time += step_dt
        if current_time >= total_time:
            break

    # 結果の保存
    np.save(os.path.join(save_dir, "eta_history.npy"), eta_history)

    # --- ヒートマップ作成 ---
    plt.figure(figsize=(9, 7))
    # x軸: km単位
    x_axis = np.arange(nx) * dx / 1000.0
    # t軸: 実際に計算したステップ数分（nt）の時間を生成
    t_axis = np.arange(nt) * dt

    X, T = np.meshgrid(x_axis, t_axis)

    heatmap = plt.pcolormesh(X, T, eta_history, cmap='jet',
                             vmin=-3, vmax=3, shading='auto')
    cbar = plt.colorbar(heatmap)
    cbar.set_label('Wave Height (m)')
    plt.xlabel('Distance from Coast (km)')
    plt.ylabel('Time (s)')
    plt.title('Space-Time Plot of Tsunami ($\\eta$)')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "eta_heatmap.png"))
    print(f"Results saved to {save_dir}")


if __name__ == "__main__":
    # パス等は環境に合わせて調整してください
    eta0_path = "../../data/wave_height_initial.csv"
    H_path = "../../data/bathemetry.csv"

    # ファイルが存在する場合のみ実行
    try:
        eta0 = np.loadtxt(eta0_path, delimiter=",", skiprows=1, usecols=1)
        H_minus = np.loadtxt(H_path, delimiter=",", skiprows=1, usecols=1)
        run_shallow_water(eta0, H_minus)
    except Exception as e:
        print(f"Error: {e}")
