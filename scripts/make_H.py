import numpy as np
import pandas as pd

from scipy.interpolate import pchip_interpolate


distance_points = np.array([
    0, 60, 62, 65, 70, 80, 85, 95, 100, 103,
    105, 107, 115, 120, 125, 133, 135, 138,
    140, 150, 250
])
depth_points = np.array([
    -0.21, -0.21, -0.4, -0.6, -0.55, -0.8, -1.2, -1.3,
    -2.1, -1.9, -1.65, -1.85, -1.78, -1.8, -2.0, -2.1,
    -2.3, -2.5, -2.7, -2.72, -2.73
])

# 2. 0.5km刻みの軸を作成 (0kmから249.5kmまで)
x_new = np.arange(0, 250, 0.5)

# 3. PCHIP補間（形状保持区分的3次エルミート多項式）
H_new = pchip_interpolate(distance_points, depth_points, x_new)

# 4. CSVデータの作成
df = pd.DataFrame({'Distance (km)': x_new, 'Depth (km)': H_new})
df.to_csv('bathymetry_reproduced.csv', index=False)
