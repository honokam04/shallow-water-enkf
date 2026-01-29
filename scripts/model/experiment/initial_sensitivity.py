import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.abspath(os.path.join(current_dir, '..', 'model'))
sys.path.append(model_dir)

from shallow_water import step_forward


def run_noisy_overlay_animation():
    # パラメータ設定
    g, dx, CFL, total_time = 9.81, 500.0, 0.1, 2500.0
    save_dir = "../../../result/results_stability"
    os.makedirs(save_dir, exist_ok=True)

    # データの読み込み
    H_raw = np.loadtxt("../../../data/bathemetry.csv",
                       delimiter=",", skiprows=1, usecols=1
                       )
    H = np.abs(H_raw) * 1000.0
    nx = len(H)
    x_km = np.arange(nx) * dx / 1000.0
    dt = CFL * dx / np.sqrt(g * np.max(H))
    nt = int(total_time / dt)

    # 初期条件に5%のノイズを加える
    eta0 = np.loadtxt("../../../data/wave_height_initial.csv",
                      delimiter=",", skiprows=1, usecols=1
                      )
    u0 = np.zeros(nx + 1)

    # ケースA: オリジナル
    e_a, u_a = eta0.copy(), u0.copy()

    # ケースB: 全体に最大振幅の5%程度のノイズを付与
    # 標準偏差を最大波高の5%に設定
    noise_level = 0.05 * np.max(np.abs(eta0))
    e_b = eta0.copy() + np.random.normal(0, noise_level, nx)
    u_b = u0.copy()

    # アニメーション設定
    fig, ax = plt.subplots(figsize=(10, 6))
    frames = []

    print(f"Calculating noisy trajectories... Noise Level: {noise_level:.3f}m")
    for t in range(nt):
        e_a, u_a, _ = step_forward(e_a, u_a, H, nx, g, dx, CFL)
        e_b, u_b, _ = step_forward(e_b, u_b, H, nx, g, dx, CFL)

        if t % 20 == 0:
            current_time = t * dt
            # 真値を黒、ノイズ入りを赤でプロット
            line_a, = ax.plot(x_km, e_a, color='black', lw=2,
                              label='True Wave', alpha=0.8
                              )
            line_b, = ax.plot(x_km, e_b, color='red', lw=1,
                              label='Noisy Wave (5% error)', alpha=0.6
                              )

            title = ax.text(0.5, 1.05,
                            f'Noise Propagation Test | Time: {current_time:.1f}s',
                            transform=ax.transAxes, ha='center', fontsize=12
                            )

            if len(frames) == 0:
                ax.set_xlabel('Distance from Coast (km)')
                ax.set_ylabel('Wave Height (m)')
                ax.set_ylim(-1.5, 3.5)
                ax.legend(loc='upper right')
                ax.grid(True, alpha=0.2)

            frames.append([line_a, line_b, title])

    # --- 4. 保存 ---
    print("Saving GIF...")
    ani = animation.ArtistAnimation(fig, frames, interval=100, blit=True)
    ani.save(f"{save_dir}/wave_noise_test.gif", writer='pillow')
    print(f"Done!")


if __name__ == "__main__":
    run_noisy_overlay_animation()
