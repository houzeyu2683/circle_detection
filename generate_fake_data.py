import csv
import os
import cv2
import numpy


def generate_fake_image(
    image_width: int = 800,
    image_height: int = 600,
    circle_radius: float = 25,
    grid_pitch: float = 80,
    position_noise: float = 3,
    random_seed: int = 0,
) -> tuple:
    """
    產生一張模擬 SEM 的灰階圖（圓排成蜂巢狀），回傳 (fake_image, true_circles)。
        fake_image    灰階圖 (高, 寬)
        true_circles  每個圓的標準答案 [[x, y, 半徑], ...]
    參數：
        image_width, image_height  圖片大小
        circle_radius              圓的平均半徑（像素）
        grid_pitch                 相鄰圓心的平均距離，要大於 2 × 半徑，圓才不會連在一起
        position_noise             每個圓心隨機偏移的程度（標準差，像素），
                                   越大，鄰居之間的距離差異越大
        random_seed                亂數種子，同一個數字每次產生的圖都一樣
    """

    random_generator = numpy.random.default_rng(random_seed)

    # 1. 背景：中間偏暗的灰色，左邊到右邊亮度慢慢變化（模仿亮度不均）
    brightness_gradient = numpy.linspace(-15, 15, image_width)[None, :]
    background_image = numpy.full(
        (image_height, image_width), 90.0
    ) + brightness_gradient
    fake_image = background_image.astype(numpy.float32)

    # 2. 圓排成蜂巢狀：每個圓的位置和大小都加一點隨機的偏差
    # 排與排的距離是 grid_pitch × √3 / 2，奇數排往右偏移半個 grid_pitch，
    # 這樣每個圓周圍的 6 個鄰居，距離剛好都是 grid_pitch
    true_circles = []
    row_spacing = grid_pitch * numpy.sqrt(3) / 2
    row_positions = numpy.arange(grid_pitch / 2, image_height, row_spacing)
    for row_index, row_position in enumerate(row_positions):
        row_offset = (grid_pitch / 2) * (row_index % 2)
        column_positions = numpy.arange(
            grid_pitch / 2 + row_offset, image_width, grid_pitch
        )
        for column_position in column_positions:
            center_x = column_position + random_generator.normal(
                0, position_noise
            )
            center_y = row_position + random_generator.normal(
                0, position_noise
            )
            radius_value = circle_radius * random_generator.uniform(0.95, 1.05)
            true_circles.append([center_x, center_y, radius_value])

    # 3. 畫圓：裡面是暗的洞，邊緣一圈亮邊（模仿 SEM 的邊緣效應）
    # cv2 畫圖只接受整數，先放大 16 倍畫，再用 shift=4 縮回來，邊緣比較準
    for center_x, center_y, radius_value in true_circles:
        drawing_center = (int(round(center_x * 16)), int(round(center_y * 16)))
        drawing_radius = int(round(radius_value * 16))
        cv2.circle(
            fake_image, drawing_center, drawing_radius,
            50, -1, cv2.LINE_AA, 4,
        )
        cv2.circle(
            fake_image, drawing_center, drawing_radius,
            200, 3, cv2.LINE_AA, 4,
        )

    # 4. 稍微模糊，讓邊緣不要太銳利，再加上顆粒狀的雜訊
    fake_image = cv2.GaussianBlur(fake_image, (0, 0), 1.2)
    noise_image = random_generator.normal(0, 12, fake_image.shape)
    fake_image = numpy.clip(fake_image + noise_image, 0, 255)
    fake_image = fake_image.astype(numpy.uint8)

    return fake_image, true_circles


def save_fake_data(
    fake_image: numpy.ndarray,
    true_circles: list,
    file_stem: str,
    data_folder: str = './Data',
) -> bool:
    """
    把圖存成 <file_stem>.jpg，標準答案存成 <file_stem>.csv。
    csv 的欄位：index, x, y, radius
    """

    os.makedirs(data_folder, exist_ok=True)
    image_path = os.path.join(data_folder, f'{file_stem}.jpg')
    answer_path = os.path.join(data_folder, f'{file_stem}.csv')

    is_saved = cv2.imwrite(image_path, fake_image)
    with open(answer_path, 'w', newline='', encoding='utf-8') as answer_file:
        answer_writer = csv.writer(answer_file)
        answer_writer.writerow(['index', 'x', 'y', 'radius'])
        for circle_index, (center_x, center_y, radius_value) in enumerate(
            true_circles
        ):
            answer_writer.writerow([
                circle_index,
                round(center_x, 2),
                round(center_y, 2),
                round(radius_value, 2),
            ])

    return is_saved


if __name__=='__main__':

    # 第 1 張：小圓、排得比較密
    fake_image, true_circles = generate_fake_image(
        circle_radius=25, grid_pitch=80, position_noise=3, random_seed=1,
    )
    save_fake_data(fake_image, true_circles, 'fake_sem_01')
    print('fake_sem_01：', len(true_circles), '個圓')

    # 第 2 張：大圓、排得比較鬆
    fake_image, true_circles = generate_fake_image(
        circle_radius=45, grid_pitch=130, position_noise=5, random_seed=2,
    )
    save_fake_data(fake_image, true_circles, 'fake_sem_02')
    print('fake_sem_02：', len(true_circles), '個圓')
