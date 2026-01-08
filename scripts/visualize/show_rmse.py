import os
import numpy as np
from PIL import Image, ImageChops


def trim_image(img):
    bg = Image.new(img.mode, img.size, img.getpixel((0, 0)))
    diff = ImageChops.difference(img, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return img.crop(bbox)
    return img


def merge_rmse_optimized():
    result_dir = "../../results"
    n_list = [2, 5, 10, 15, 20, 30, 50, 70, 100]

    images = []
    for n in n_list:
        filename = f"dx=60_dt=3.0_N={n}_rmse.png"
        path = os.path.join(result_dir, filename)
        if os.path.exists(path):
            img = Image.open(path).convert("RGB")
            # 余白をカット
            img = trim_image(img)
            images.append(img)
        else:
            print(f"Warning: {filename} not found.")

    if not images:
        return

    w, h = images[0].size

    # 3x3のキャンバスを作成
    padding = 20
    canvas_w = w * 3 + padding * 2
    canvas_h = h * 3 + padding * 2
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))

    for i, img in enumerate(images):
        # 3x3にリサイズ
        img = img.resize((w, h), Image.LANCZOS)
        x = (i % 3) * (w + padding)
        y = (i // 3) * (h + padding)
        canvas.paste(img, (x, y))

    # 保存
    save_path = os.path.join(result_dir, "merged_rmse.png")
    canvas.save(save_path)
    canvas.show()
    print(f"統合完了: {save_path}")


if __name__ == "__main__":
    merge_rmse_optimized()
