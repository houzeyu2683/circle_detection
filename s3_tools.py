import cv2
import numpy

def get_gradient(
    gray_image: numpy.ndarray
) -> tuple:
    """
    算出每個像素的亮度變化，回傳 (gradient_x, gradient_y, gradient_magnitude)。
        gradient_x          往右走，亮度變化多少（變亮 = 正，變暗 = 負）
        gradient_y          往下走，亮度變化多少
        gradient_magnitude  合起來的變化有多大，也就是邊緣強度 √(x² + y²)
    三個陣列的形狀都跟 gray_image 一樣。
        gray_image  灰階圖，通常是模糊過的
    """

    # CV_32F：結果可以存小數和負數
    # 1, 0：算 x 方向（往右）；0, 1：算 y 方向（往下）
    gradient_x = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0)
    gradient_y = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1)
    gradient_magnitude = cv2.magnitude(gradient_x, gradient_y)
    return gradient_x, gradient_y, gradient_magnitude
