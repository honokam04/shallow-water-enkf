import numpy as np


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
