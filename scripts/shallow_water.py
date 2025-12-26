# 1次元浅水波モデル(1ステップ)
def step_forward(eta, u, H, nx, g, dx, dt, c_out):
    # 運動方程式
    u_new = u.copy()
    eta_new = eta.copy()

    u_new[1:nx] = u[1:nx] - (g * dt / dx) * (eta[1:nx] - eta[0:nx-1])
    u_new[0] = 0.0    # 沿岸境界
    u_new[-1] = u[-1] - (c_out * dt / dx) * (u[-1] - u[-2])  # 放射境界

    # 連続の式
    eta_new[0:nx-1] = eta[0:nx-1] - (
        (dt / dx) * H[0:nx-1] * (u_new[1:nx] - u_new[0:nx-1])
    )
    # 対流流出条件
    eta_new[-1] = eta[-1] - (
        (c_out * dt / dx) * (eta[-1] - eta[-2])
    )

    return eta_new, u_new
