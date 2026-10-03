import cv2
import numpy


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

    pass

if __name__=='__main__':

    image_path = './Data/sample_03.jpg'
    picture = Picture(image_path)
    color_image = picture.read_image()
    print('圖片大小（高, 寬, 顏色數）：', color_image.shape)
    pass
