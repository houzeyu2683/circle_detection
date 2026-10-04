import cv2
import numpy
import typing
import tqdm
import s1_tools

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

    image_path = './Data/sample_05.jpg'
    picture = Picture(image_path)
    picture.read_color_image()
    picture.read_gray_image()

    group_circles = []
    for attempt_size in [150, 300, 600]:
        # 給定一個尺度，找出這個尺度的候選圓（原圖座標）
        candidate_circles = s1_tools.get_candidate_circles(image_path, attempt_size)
        group_circles += candidate_circles
        print(len(candidate_circles))
        continue

    selected_circles = s1_tools.get_selected_circles(group_circles)
    print(len(selected_circles))

    # 把選上的圓畫在彩色原圖上存檔，用眼睛確認結果
    save_path = './check.jpg'
    is_saved = picture.draw_circles(selected_circles, save_path)
    print('存檔成功：', is_saved, save_path)
    pass


