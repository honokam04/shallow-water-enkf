import numpy as np
import os
import pandas as pd

from scipy.interpolate import pchip_interpolate


def create_csv() -> pd.DataFrame:
    # 1. グラフから抽出した主要な制御点 (Distance, Wave Height)
    dist_pts = np.array([
        0, 10, 30, 50, 60, 61.5, 63, 65,
        75, 80, 88, 95, 103, 105, 107, 110,
        115, 120, 125, 130, 135, 138, 145, 150, 250
    ])
    wave_pts = np.array([
        -1.2, -1.6, -0.5, 1.6, 1.6, 3.7, 1.2, 1.7,
        2.5, 2.3, 1.6, 2.6, 0.2, 1.7, 0.7, 0.7, 1.2,
        1.1, 1.5, 1.5, 2.0, 0.5, 0.05, 0, 0
    ])

    # 2. 0.5km刻みの軸を作成 (0kmから249.5kmまで)
    x_new = np.arange(0, 250, 0.5)

    # 3. PCHIP補間（形状保持区分的3次エルミート多項式）
    eta0_new = pchip_interpolate(dist_pts, wave_pts, x_new)

    # 4. データフレームの作成
    df = pd.DataFrame({'Distance (km)': x_new,
                       'Wave Height (m)': eta0_new})
    return df


def main():
    csv_path = '../data'

    if os.path.isdir(csv_path):
        pass
    else:
        os.mkdir(csv_path)

    df = create_csv()
    df.to_csv(csv_path + '/wave_height_initial.csv', index=False)


if __name__ == '__main__':
    main()
