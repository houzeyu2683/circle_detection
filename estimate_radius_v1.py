import cv2
import numpy
import os
import tools

class Picture:

    def __init__(self, checkpoint_path: str = '.tmp/') -> None:
        """checkpoint_path: 中間處理的暫存檔案保存位置。"""

        self.checkpoint_path = checkpoint_path
        return

    def read_image(self, file_path: str) -> bool:
        """讀取圖片檔案"""

        image = {}
        image['colorful'] = cv2.imread(file_path)
        image['gray'] = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        self.image = image
        return True

    def get_scale_ratio(self, size_boundary: int) -> float:
        """獲得縮小比例"""

        shape = numpy.array(self.image['gray']).shape
        scale_ratio = size_boundary / max(shape[:2])
        return scale_ratio

    def get_gray_image(self, scale_ratio: float = 1.0) -> numpy.ndarray:
        """
        獲得灰階的圖片陣列，並依照縮放比例縮放。
        """

        if scale_ratio == 1.0:

            gray_image = numpy.array(self.image['gray']).copy()
            return gray_image

        # 縮小和放大用不同的方法，畫質比較好：
        #   INTER_AREA：縮小時把好幾個像素平均成一個，不會有鋸齒或雜點
        #   INTER_CUBIC：放大時用周圍 4×4 個像素算出新像素，邊緣比較平滑
        if scale_ratio < 1.0:

            resize_method = cv2.INTER_AREA
            pass

        else:
            
            resize_method = cv2.INTER_CUBIC
            pass

        gray_image = cv2.resize(
            self.image['gray'], 
            None,
            fx=scale_ratio, 
            fy=scale_ratio,
            interpolation=resize_method,
        )
        return gray_image

    pass

if __name__=='__main__':

    file_path = './sample_01.jpg'
    checkpoint_path = '.tmp2'
    picture = Picture(checkpoint_path)
    picture.read_image(file_path)

    # ================================================================
    # v1 的流程：估計 r，並給出一批建議的圓
    # 跟 01_estimate_radius.py 比，拿掉了快篩和 refine（r 只差不到 1%）
    # 輸出：
    #   estimated_radius    r（原圖像素）
    #   suggested_circles   [[x, y, r, score], ...]（原圖座標）
    # ================================================================

    # ---- 步驟 1：金字塔，每一層各找一次圓 ----
    # 準備一個空的 list，裝三層找到的圓：merged_circles = []
    # for size_boundary in (600, 300, 150):
    #
    #     1-1 縮放
    #         scale_ratio = picture.get_scale_ratio(size_boundary)
    #         scale_ratio > 1 → 圖比這層還小，continue 跳過
    #         gray_image = picture.get_gray_image(scale_ratio)
    #
    #     1-2 算邊緣（模糊 1.5）
    #         → (gradient, magnitude)
    #         01 裡的 get_gray_edges
    #
    #     1-3 Hough 猜圓（用 01 的預設值，什麼大小都找）
    #         → [[x, y, r], ...]
    #         01 裡的 get_circle_candidates
    #
    #     1-4 打分數（72 個點，強邊緣門檻 0.8）
    #         → [score, ...]
    #         01 裡的 get_circle_scores，用預設值
    #
    #     1-5 過濾：分數 ≥ 0.7 才留下
    #         → [[x, y, r, score], ...]
    #         01 裡的 get_filtered_circles
    #
    #     1-6 NMS：同一顆圓只留一個，並除以 scale_ratio 換回原圖座標
    #         → [[x, y, r, score], ...]（原圖座標）
    #         01 裡的 get_unique_circles(..., scale_ratio=scale_ratio)
    #
    #     1-7 加進 merged_circles

    # ---- 步驟 2：三層合併，再 NMS 一次 ----
    # 同一顆圓可能在好幾層都被找到
    # 01 裡的 get_unique_circles，不傳 scale_ratio（已經是原圖座標）

    # ---- 步驟 3：估計 r ----
    # 圓少於 3 個 → 估不出 r，停止
    # 面積加權的半徑分布，最高點 = r
    # 01 裡的 get_estimated_radius

    # ---- 步驟 4：建議的圓 ----
    # 從步驟 2 的圓裡，只留半徑在 0.6r ～ 1.4r 之間的
    # （跟第二階段 Hough 找圓的範圍一樣，會刪掉細胞裡的小團塊）
    # → suggested_circles

    # ---- 步驟 5：畫圖存檔 ----
    # 把 suggested_circles 畫在彩色原圖上，存到 checkpoint_path
    # 01 裡的 save_image

    pass