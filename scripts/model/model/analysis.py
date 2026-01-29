import numpy as np


# 津波到達時刻、最大波高、最大波到達時刻の算出
def analyze_and_compare_tsunami(eta_true, eta_analysis, dt, threshold=0.2):

    def get_coastal_stats(history, label):
        # 沿岸(x=0)の時系列のみを抽出
        coastal_eta = history[:, 0]

        # 1. 津波到達時刻 (左端で threshold を初めて超える時刻)
        arrival_idx = np.where(coastal_eta > threshold)[0]
        arrival_time = arrival_idx[0] * dt if len(arrival_idx) > 0 else None

        # 2. 沿岸での最大波高とその到達時刻
        max_idx_t = np.argmax(coastal_eta)
        max_height = coastal_eta[max_idx_t]
        max_arrival_time = max_idx_t * dt

        return {
            "label": label,
            "arrival_time": arrival_time,
            "max_height": max_height,
            "max_time": max_arrival_time
        }

    stats_true = get_coastal_stats(eta_true, "True (真値)")
    stats_ana = get_coastal_stats(eta_analysis, "Analysis (同化値)")

    # 結果の表示
    print("\n" + "="*55)
    print(f"{'項目 (沿岸 x=0)':<20} | {'True (真値)':<15} | {'Analysis (同化値)':<15}")
    print("-"*55)

    # 到達時刻の比較
    t_true = f"{stats_true['arrival_time']:.2f}s" if stats_true['arrival_time'] is not None else "N/A"
    t_ana = f"{stats_ana['arrival_time']:.2f}s" if stats_ana['arrival_time'] is not None else "N/A"
    print(f"{'津波到達時刻':<20} | {t_true:<15} | {t_ana:<15}")

    # 最大波高の比較
    print(f"{'最大波高':<20} | {stats_true['max_height']:>14.3f}m | {stats_ana['max_height']:>14.3f}m")

    # 最大波到達時刻の比較
    print(f"{'最大波到達時刻':<20} | {stats_true['max_time']:>14.2f}s | {stats_ana['max_time']:>14.2f}s")
    print("="*55)

    return stats_true, stats_ana
