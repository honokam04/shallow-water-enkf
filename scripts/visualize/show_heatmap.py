import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def merge_rmse_images():
    # 設定
    result_dir = "../../results"
    # アンサンブル数 N のリスト
    n_list = [2, 5, 10, 15, 20, 30, 50, 70, 100]

    # 描画設定 (3行 x 3列)
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    plt.subplots_adjust(wspace=0.2, hspace=0.3)

    found_files = 0
    for i, n in enumerate(n_list):
        row = i // 3
        col = i % 3

        filename = f"dx=60_dt=3.0_N={n}.png"
        file_path = os.path.join(result_dir, filename)

        if os.path.exists(file_path):
            img = mpimg.imread(file_path)
            axes[row, col].imshow(img)
            axes[row, col].set_title(f"Ensemble Size N = {n}",
                                     fontsize=14, fontweight='bold')
            found_files += 1
        else:
            axes[row, col].text(0.5, 0.5, f"File Not Found:\n{filename}",
                                ha='center', va='center', color='red')

        # 軸を非表示にする
        axes[row, col].axis('off')

    # 全体のタイトル
    plt.suptitle("Comparison of Heatmaps by Ensemble Size (N)",
                 fontsize=20, y=0.95)

    # 保存
    save_name = os.path.join(result_dir, "merged_heatmap.png")
    plt.savefig(save_name, bbox_inches='tight', dpi=150)
    plt.show()

    print(f"統合完了: {save_name} (見つかったファイル: {found_files}/9)")


if __name__ == "__main__":
    merge_rmse_images()
