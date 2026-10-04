import cv2
import numpy
import typing
import s3_tools

class Picture:

    def __init__(self, image_path: str) -> None:

        self.image_path = image_path
        return

    def read_color_image(self) -> bool:
        """讀取彩色圖片，回傳圖片陣列，形狀是 (高, 寬, 3)。"""

        color_image = cv2.imread(self.image_path)
        # cv2.imread 讀不到檔案時不會報錯，只會回傳 None，所以自己檢查
        if color_image is None:
            return False

        if color_image.shape[-1] != 3:
            return False

        self.color_image = color_image
        return True

    def read_gray_image(self) -> bool:
        """讀取灰階圖片，存到 self.gray_image，形狀是 (高, 寬)。"""

        # cv2.IMREAD_GRAYSCALE：讀進來時直接轉成灰階，每個像素只有一個亮度值
        gray_image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        # cv2.imread 讀不到檔案時不會報錯，只會回傳 None，所以自己檢查
        if gray_image is None:
            return False

        # 灰階圖只有高、寬兩個維度，沒有顏色那一維
        if gray_image.ndim != 2:
            return False

        self.gray_image = gray_image
        return True

    def get_image_shape(
        self, image_mode: typing.Literal['color', 'gray'] = 'gray'
    ) -> tuple:

        if image_mode == 'gray':
            image_shape = self.gray_image.shape
            return image_shape

        image_shape = self.color_image.shape
        return image_shape

    def draw_circles(self, target_circles: list, save_path: str) -> bool:
        """把圓畫在彩色原圖上，存到 save_path。"""

        # 畫在複製品上，原本的 self.color_image 不會被改到
        draw_image = self.color_image.copy()

        for target_circle in target_circles:
            # 只用前三個值，有沒有分數都可以
            center_x, center_y, circle_radius = target_circle[:3]
            # cv2.circle 只接受整數
            circle_center = (int(round(center_x)), int(round(center_y)))
            # 紅色圓周 (B, G, R) = (0, 0, 255)，線條粗細 2
            cv2.circle(
                draw_image, circle_center, int(round(circle_radius)),
                (0, 0, 255), 2,
            )
            # 綠色圓心點，粗細 -1 = 塗滿
            cv2.circle(draw_image, circle_center, 3, (0, 255, 0), -1)

        # cv2.imwrite 存檔成功回傳 True，失敗回傳 False
        is_saved = cv2.imwrite(save_path, draw_image)
        return is_saved

    pass

if __name__=='__main__':

    image_path = './Data/sample_03.jpg'
    picture = Picture(image_path)
    picture.read_color_image()
    picture.read_gray_image()

    # 假設我們找到圓，這些圓也被還原大小了。
    detected_circles = [
        [255.375, 266.625, 91.79999542236328, 1.0],
        [437.625, 298.125, 89.7750015258789, 0.9861111111111112],
        [509.625, 133.875, 86.8499984741211, 1.0],
        [689.625, 199.125, 96.07500457763672, 0.875],
        # [91.125, 401.625, 76.04999542236328, 0.6666666666666666],
        # [712.125, 439.875, 65.92499542236328, 0.7083333333333334],
        # [331.875, 419.625, 61.42499923706055, 0.6805555555555556],
    ]
    print(len(detected_circles))

    # 1. 在原圖上模糊、算梯度
    # 模糊取圓半徑的 1/20，半徑用 detected_circles 的中位數
    median_radius = numpy.median(
        [detected_circle[2] for detected_circle in detected_circles]
    )
    blur_sigma = median_radius / 20
    blur_source = cv2.GaussianBlur(picture.gray_image, (0, 0), blur_sigma)
    gradient_x, gradient_y, gradient_magnitude = s3_tools.get_gradient(
        blur_source
    )
    print(median_radius, blur_sigma, gradient_magnitude.shape)

    # 2. 拿一個圓練習：左下那個偏掉的圓
    center_x, center_y, circle_radius, _ = detected_circles[0]

    # 90 條射線的方向，直著排，形狀 (90, 1)
    ray_angles = numpy.linspace(0, 2 * numpy.pi, 90, endpoint=False)
    direction_x = numpy.cos(ray_angles)[:, None]
    direction_y = numpy.sin(ray_angles)[:, None]

    # 每條射線上 31 個點，離圓心 0.85r ～ 1.15r，橫著排，形狀 (1, 11)
    sample_radius = circle_radius * numpy.linspace(0.90, 1.10, 11)[None, :]

    # (90, 1) × (1, 31) → (90, 31)
    sample_x = numpy.round(center_x + sample_radius * direction_x).astype(int)
    sample_y = numpy.round(center_y + sample_radius * direction_y).astype(int)
    print(sample_x.shape)

    # 3. 每條射線找邊緣最強的點
    image_height, image_width = gradient_magnitude.shape
    # 每條射線的 31 個點「全部」在圖裡，才用這條射線 → (90,)
    ray_inside = (
        (sample_x >= 0) & (sample_x < image_width)
        & (sample_y >= 0) & (sample_y < image_height)
    ).all(axis=1)
    # 先把圖外的座標拉回邊界，只是為了下一行取值不出錯
    sample_x = numpy.clip(sample_x, 0, image_width - 1)
    sample_y = numpy.clip(sample_y, 0, image_height - 1)

    # 每個取樣點的邊緣強度 → (90, 31)
    sample_magnitude = gradient_magnitude[sample_y, sample_x]
    # 每條射線上，最強的是第幾個點 → (90,)
    strongest_index = numpy.argmax(sample_magnitude, axis=1)

    # 取出每條射線最強那一點的座標，只留在圖裡的射線
    ray_index = numpy.arange(90)
    edge_x = sample_x[ray_index, strongest_index][ray_inside]
    edge_y = sample_y[ray_index, strongest_index][ray_inside]
    print(len(edge_x))

    # 4. 用外框點擬合一個圓（最小平方法）
    # 先丟掉最弱的 25%：那些射線可能沒碰到真的外框，
    # 只是在雜訊裡挑了相對最強的一點
    edge_magnitude = sample_magnitude[ray_index, strongest_index][ray_inside]
    is_strong = edge_magnitude > numpy.percentile(edge_magnitude, 25)
    edge_x = edge_x[is_strong].astype(float)
    edge_y = edge_y[is_strong].astype(float)

    # 圓的方程式整理成 [2x, 2y, 1] · [a, b, c] = x² + y²，每個點一條方程式
    # lstsq 找出讓所有方程式整體誤差最小的 a, b, c
    fit_matrix = numpy.column_stack(
        [2 * edge_x, 2 * edge_y, numpy.ones(len(edge_x))]
    )
    fit_target = edge_x ** 2 + edge_y ** 2
    fit_x, fit_y, fit_c = numpy.linalg.lstsq(
        fit_matrix, fit_target, rcond=None
    )[0]
    # 半徑由 c = R² - a² - b² 反推
    fit_radius = numpy.sqrt(fit_c + fit_x ** 2 + fit_y ** 2)
    print(fit_x, fit_y, fit_radius)

    pass
