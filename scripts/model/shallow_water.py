import numpy as np


def step_forward(eta, u, H, nx, g, dx, CFL):
    # 関数内で dt を決定（物理的安定性を担保）
    dt = CFL * dx / np.sqrt(g * np.max(H))
    c_out = np.sqrt(g * H[-1])

    u_new = u.copy()
    eta_new = eta.copy()

    # 1. 運動方程式
    u_new[1:nx] = u[1:nx] - (g * dt / dx) * (eta[1:nx] - eta[0:nx-1])
    u_new[0] = 0.0    # 沿岸
    u_new[nx] = u[nx] - (c_out * dt / dx) * (u[nx] - u[nx-1])  # 放射

    # 2. 連続の式
    eta_new[0:nx] = eta[0:nx] - (
        (dt / dx) * H[0:nx] * (u_new[1:nx+1] - u_new[0:nx])
        )

    # 3. 水位の放射境界
    eta_new[-1] = eta[-1] - (c_out * dt / dx) * (eta[-1] - eta[-2])

    return eta_new, u_new, dt
