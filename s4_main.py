import csv
import cv2
import numpy
import typing
import s4_tools

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

    def draw_neighbors(
        self,
        target_circles: list,
        neighbor_pairs: list,
        save_path: str,
    ) -> bool:
        """
        把相鄰的兩個圓心用黃線連起來，畫在彩色原圖上，存到 save_path。
            target_circles  圓 [[x, y, 半徑, ...], ...]
            neighbor_pairs  鄰居 [[i, j, 距離], ...]，i、j 是 target_circles 的編號
        """

        # 畫在複製品上，原本的 self.color_image 不會被改到
        draw_image = self.color_image.copy()

        # 黃線 (B, G, R) = (0, 255, 255)：連接每一對鄰居的圓心
        for first_index, second_index, pair_distance in neighbor_pairs:
            first_center = (
                int(round(target_circles[first_index][0])),
                int(round(target_circles[first_index][1])),
            )
            second_center = (
                int(round(target_circles[second_index][0])),
                int(round(target_circles[second_index][1])),
            )
            cv2.line(draw_image, first_center, second_center, (0, 255, 255), 1)

        # 綠色圓心點，粗細 -1 = 塗滿
        for target_circle in target_circles:
            circle_center = (
                int(round(target_circle[0])), int(round(target_circle[1]))
            )
            cv2.circle(draw_image, circle_center, 3, (0, 255, 0), -1)

        # cv2.imwrite 存檔成功回傳 True，失敗回傳 False
        is_saved = cv2.imwrite(save_path, draw_image)
        return is_saved

    pass

if __name__=='__main__':

    image_path = './Data/fake_sem_01.jpg'
    picture = Picture(image_path)
    picture.read_color_image()
    picture.read_gray_image()

    # 輸入的圓：先用模擬資料的標準答案（csv），格式 [x, y, 半徑]
    answer_path = './Data/fake_sem_01.csv'
    target_circles = []
    with open(answer_path, encoding='utf-8') as answer_file:
        for answer_row in csv.DictReader(answer_file):
            target_circles.append([
                float(answer_row['x']),
                float(answer_row['y']),
                float(answer_row['radius']),
            ])
    print(len(target_circles))

    # 1. 距離表：每兩個圓心之間的距離
    distance_table = s4_tools.get_distance_table(target_circles)

    # 2. 找鄰居：每個圓找最近的 6 個（蜂巢排列），再刪掉距離太遠的配對
    neighbor_pairs = s4_tools.get_neighbor_pairs(
        distance_table, neighbor_count=6
    )

    # 3. 統計
    distance_summary = s4_tools.get_distance_summary(neighbor_pairs)
    for summary_name, summary_value in distance_summary.items():
        print(summary_name, summary_value)

    # 4. 把鄰居連線畫出來，用眼睛確認
    save_path = './check_s4.jpg'
    is_saved = picture.draw_neighbors(
        target_circles, neighbor_pairs, save_path
    )
    print('存檔成功：', is_saved, save_path)

    pass
