import numpy as np
import matplotlib.pyplot as plt
import os


def run_shallow_water(eta0, H_new, dx=500.0, g=9.81,
                      total_time=3000.0, CFL=0.8, save_dir="../results"):

    # ディレクトリ作成
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 水深と初期条件
    H = np.abs(H_new) * 1000.0     # H_new (km) -> 水深 H (m)
    eta_initial = eta0.copy()      # 初期水位 η(x, 0)
    nx = len(H)           # モデルの格子点数

    # CFL条件から時間刻み dt を計算
    dt = CFL * dx / np.sqrt(g * np.max(H))
    nt = int(total_time / dt)
    print(f"dx={dx} m, dt={dt:.3f} s, nt={nt} steps")  # 時間ステップ総数

    # 配列初期化
    eta = eta_initial.copy()   # eta[i]は格子点x[i]の水位
    u = np.zeros(nx + 1)       # スタガード格子での流速
    eta_history = np.zeros((nt, nx))

    # 数値計算(スタガード格子による Leap-frog 法)
    for t_step in range(nt):
        # (A) 運動方程式：流速 u の更新
        for i in range(1, nx):
            u[i] -= (g * dt / dx) * (eta[i] - eta[i-1])

        # 境界条件
        u[0] = 0          # q = U*H かつ q(0,t)=0 すなわち u(0,t)=0
        u[-1] = u[-2]     # 非反射条件（1階の外挿）
        eta[-1] = eta[-2]

        # (B) 連続の式：水位 η の更新
        for i in range(0, nx-1):
            eta[i] -= (dt / dx) * H[i] * (u[i+1] - u[i])

        # 履歴保存
        eta_history[t_step, :] = eta

    # 結果の保存
    np.save(os.path.join(save_dir, "eta_history.npy"), eta_history)

    # ヒートマップ作成
    plt.figure(figsize=(9, 7))
    x_axis = np.linspace(0, dx*(nx-1)/1000, nx)    # km に変換
    t_axis = np.linspace(0, total_time, nt)
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
    plt.show()

    print("Simulation completed. Results saved in folder:", save_dir)


if __name__ == "__main__":
    eta0_path = "../data/wave_height_initial.csv"
    H_path = "../data/bathemetry.csv"

    eta0_new = np.loadtxt(eta0_path, delimiter=",", skiprows=1, usecols=1)
    H_new = np.loadtxt(H_path, delimiter=",", skiprows=1, usecols=1)

    run_shallow_water(eta0_new, H_new)
