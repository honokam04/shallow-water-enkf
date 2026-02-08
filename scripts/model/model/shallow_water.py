import numpy as np


# 時間微分値 (d_eta/dt, du/dt) を計算する
def get_derivative(eta, u, H, nx, g, dx):
    d_u = np.zeros_like(u)
    d_eta = np.zeros_like(eta)
    c_out = np.sqrt(g * H[-1])

    # 1. 運動方程式の右辺: du/dt = -g * d_eta/dx
    # 沿岸(x=0)の流速は常に0なので微分値も0
    d_u[1:nx] = - (g / dx) * (eta[1:nx] - eta[0:nx-1])
    d_u[0] = 0.0
    # 放射境界条件
    d_u[nx] = - (c_out / dx) * (u[nx] - u[nx-1])

    # 2. 連続の式の右辺: d_eta/dt = -H * du/dx
    d_eta[0:nx] = - (H[0:nx] / dx) * (u[1:nx+1] - u[0:nx])
    # 水位の放射境界
    d_eta[-1] = - (c_out / dx) * (eta[-1] - eta[-2])

    return d_eta, d_u


# 4次ルンゲ=クッタ法 (RK4) による1ステップ更新
def step_forward_rk4(eta, u, H, nx, g, dx, CFL):
    dt = CFL * dx / np.sqrt(g * np.max(H))

    # k1 (現在の値での傾き)
    k1_eta, k1_u = get_derivative(eta, u, H, nx, g, dx)

    # k2 (k1を使って半分進んだ地点での傾き)
    k2_eta, k2_u = get_derivative(eta + 0.5 * dt * k1_eta,
                                  u + 0.5 * dt * k1_u, H, nx, g, dx)

    # k3 (k2を使って半分進んだ地点での傾き)
    k3_eta, k3_u = get_derivative(eta + 0.5 * dt * k2_eta,
                                  u + 0.5 * dt * k2_u, H, nx, g, dx)

    # k4 (k3を使って1ステップ進んだ地点での傾き)
    k4_eta, k4_u = get_derivative(eta + dt * k3_eta,
                                  u + dt * k3_u, H, nx, g, dx)

    # 4つの傾きを重み付き平均して更新
    eta_new = eta + (dt / 6.0) * (k1_eta + 2*k2_eta + 2*k3_eta + k4_eta)
    u_new = u + (dt / 6.0) * (k1_u + 2*k2_u + 2*k3_u + k4_u)

    return eta_new, u_new, dt


# 陽的オイラー法による浅水波モデル
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
