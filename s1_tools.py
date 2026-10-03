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

def get_circle_score(
    center_x: float,
    center_y: float,
    circle_radius: float,
    gradient_x: numpy.ndarray,
    gradient_y: numpy.ndarray,
    gradient_magnitude: numpy.ndarray,
    strong_threshold: float
) -> float:

    node_size = 72
    radius_epsilon = 2
    # 1. 算出 72 × 5 個點的座標（每一列是一個角度，每一欄是一圈）
    point_angles = numpy.linspace(0, 2 * numpy.pi, node_size, endpoint=False)
    # 每個角度的方向（往右 cos、往下 sin），直著排，形狀 (72, 1)
    direction_x = numpy.cos(point_angles)[:, None]
    direction_y = numpy.sin(point_angles)[:, None]
    # 5 圈的半徑橫著排：[r-2, r-1, r, r+1, r+2]，形狀 (1, 5)
    point_offsets = numpy.arange(-radius_epsilon, radius_epsilon + 1)
    point_radius = circle_radius + point_offsets[None, :]
    # (72, 1) × (1, 5) → 自動展開成 (72, 5)
    point_x = center_x + point_radius * direction_x
    point_y = center_y + point_radius * direction_y
    point_x = numpy.array(point_x).astype(int)
    point_y = numpy.array(point_y).astype(int)

    # 2. 處理圖外的點：只要有一個點在圖外，整個圓就不考慮
    gradient_height, gradient_width = gradient_magnitude.shape
    # .all()：72 × 5 個點全部在圖裡，才是 True
    point_inside = (
        numpy.array(point_x >= 0) & 
        numpy.array(point_x < gradient_width) & 
        numpy.array(point_y >= 0) & 
        numpy.array(point_y < gradient_height)
    ).all()
    if not point_inside:
        return 0.0

    # 3. 第 1 題：point_strong
    # 這裡有強邊緣嗎？注意順序是 [y, x]
    point_magnitude = gradient_magnitude[point_y, point_x]
    point_strong = point_magnitude > strong_threshold

    # 4. 第 2 題：point_radial
    # 箭頭有沒有沿著半徑方向？point_cosine 越接近 1，夾角越小
    # + 1e-6：避免箭頭長度是 0 時變成除以 0
    point_gradient_x = gradient_x[point_y, point_x]
    point_gradient_y = gradient_y[point_y, point_x]
    point_cosine = numpy.abs(
        point_gradient_x * direction_x
        + point_gradient_y * direction_y
    ) / (point_magnitude + 1e-6)
    point_radial = point_cosine > 0.8

    # 5. 算分數：circle_score
    # 兩題都對才得分
    # .any(axis=1)：每個角度的 5 圈裡任一格得分，這個角度就得分 → 72 個
    # 分數 = 得分的角度 ÷ 72
    point_hit = (point_strong & point_radial).any(axis=1)
    circle_score = float(point_hit.sum() / node_size)
    return circle_score

def get_selected_circles(
    scored_circles: list,
    overlap_ratio: float = 0.75,
    inside_ratio: float = 0.85,
) -> list:
    """
    去重複：擠在一起的圓只選最好的一個。
        scored_circles  有分數的圓 [[x, y, 半徑, 分數], ...]
        overlap_ratio   圓心距離 < overlap_ratio × (兩半徑相加) → 重疊太多
        inside_ratio    圓心距離 < inside_ratio × 較大的半徑 → 包在裡面
    回傳選上的圓，格式一樣 [[x, y, 半徑, 分數], ...]。
    """

    # 1. 排隊：依「分數 × √半徑」由大排到小
    # 分數差不多時讓大圓排前面，大圓裡的小團塊之後會因為「包在裡面」被淘汰
    sorted_circles = sorted(
        scored_circles,
        key=lambda scored_circle: (
            scored_circle[3] * numpy.sqrt(scored_circle[2])
        ),
        reverse=True,
    )

    # 2. 一個一個看，跟已經選上的圓比較，沒有衝突才選上
    selected_circles = []
    for center_x, center_y, circle_radius, circle_score in sorted_circles:
        is_conflict = False
        for selected_x, selected_y, selected_radius, _ in (
            selected_circles
        ):
            # 兩個圓心的距離 √(Δx² + Δy²)
            center_distance = numpy.hypot(
                center_x - selected_x, center_y - selected_y
            )
            # 衝突一：重疊太多
            if center_distance < overlap_ratio * (
                circle_radius + selected_radius
            ):
                is_conflict = True
                break
            # 衝突二：包在裡面
            if center_distance < inside_ratio * max(
                circle_radius, selected_radius
            ):
                is_conflict = True
                break
        if not is_conflict:
            selected_circles.append(
                [center_x, center_y, circle_radius, circle_score]
            )

    return selected_circles

def get_candidate_circles(
    image_path: str,
    attempt_size: int,
) -> list:
    """
    在一種尺寸上找圓，回傳原圖座標的候選圓 [[x, y, 半徑, 分數], ...]。
        image_path    圖片路徑
        attempt_size  把圖縮到最長邊是多少，例如 600
    一個圓都沒找到時回傳 []。
    """

    # 0. 讀灰階圖：讀不到就報錯
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray_image is None:
        raise FileNotFoundError(f'讀不到圖片：{image_path}')

    # 1. 縮放：最長邊縮到 attempt_size → gray_source
    scale_ratio = attempt_size / max(gray_image.shape)
    gray_source = cv2.resize(
        gray_image, None,
        fx=scale_ratio, fy=scale_ratio,
        interpolation=cv2.INTER_AREA,
    )

    # 2. 模糊 → blur_source
    blur_sigma = 1.5
    blur_source = cv2.GaussianBlur(gray_source, (0, 0), blur_sigma)

    # 3. 算梯度 → gradient_x, gradient_y, gradient_magnitude
    gradient_x, gradient_y, gradient_magnitude = get_gradient(blur_source)

    # 4. Hough 猜圓 → source_circles：條件放寬，寧可多猜
    source_circles = cv2.HoughCircles(
        blur_source,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=4,
        param1=80,
        param2=25,
        minRadius=6,
        maxRadius=min(blur_source.shape) // 2,
    )
    # 一個都沒猜到時，cv2 回傳 None
    if source_circles is None:
        return []

    # 5. 打分數 + 過濾 → passed_circles：只留下分數 ≥ 0.7 的
    strong_threshold = numpy.quantile(gradient_magnitude, 0.8)
    passed_circles = []
    for center_x, center_y, circle_radius in source_circles[0]:
        circle_score = get_circle_score(
            center_x, center_y, circle_radius,
            gradient_x, gradient_y, gradient_magnitude,
            strong_threshold,
        )
        if circle_score < 0.7:
            continue
        passed_circles.append(
            [center_x, center_y, circle_radius, circle_score]
        )

    # 6. 去重複 → selected_circles
    selected_circles = get_selected_circles(passed_circles)

    # 7. 換回原圖 → candidate_circles：x、y、半徑除以 scale_ratio，分數不用換
    candidate_circles = []
    for center_x, center_y, circle_radius, circle_score in selected_circles:
        candidate_circles.append([
            float(center_x / scale_ratio),
            float(center_y / scale_ratio),
            float(circle_radius / scale_ratio),
            circle_score,
        ])
        continue

    return candidate_circles
