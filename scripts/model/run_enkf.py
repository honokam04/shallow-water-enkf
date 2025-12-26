import numpy as np
import matplotlib.pyplot as plt
import os


# アンサンブルカルマンフィルタ
def analysis_step(Uhat, y_obs_history, n_obs, H, R, N):

    # 1. 予測値を観測空間に投影 (H * X_prime)
    y = y_obs_history

    # 3. 予測
    mhat = np.mean(Uhat, axis=1, keepdims=True)
    Chat = ((Uhat - mhat) @ (Uhat - mhat).T) / (N - 1)

    # 4. 観測値への摂動
    obs_noise = np.random.multivariate_normal(np.zeros(n_obs), R, N).T
    y_perturbed = y.reshape(-1, 1) + obs_noise

    # 5. カルマンゲインの計算
    d = y_perturbed - H @ Uhat
    C_HT = Chat @ H.T
    S = H @ Chat @ H.T + R
    K = C_HT @ np.linalg.inv(S)

    # 6. アンサンブルの更新
    U = Uhat + K @ d

    return U
