import numpy

def get_distance_table(
    target_circles: list
) -> numpy.ndarray:
    """
    算出每兩個圓心之間的距離，回傳 N × N 的距離表。
        target_circles  圓 [[x, y, 半徑, ...], ...]，只用前兩個值
    distance_table[i, j] = 第 i 個圓到第 j 個圓的圓心距離，對角線是 0。
    """

    # 所有圓心排成 (N, 2) 的表格，每一列是一個圓心 [x, y]
    circle_centers = numpy.array(
        [target_circle[:2] for target_circle in target_circles], dtype=float
    )
    # (N, 1, 2) - (1, N, 2) → (N, N, 2)：每個圓心減每個圓心
    center_differences = circle_centers[:, None, :] - circle_centers[None, :, :]
    # 畢氏定理 √(Δx² + Δy²) → (N, N)
    distance_table = numpy.sqrt((center_differences ** 2).sum(axis=2))
    return distance_table

def get_neighbor_pairs(
    distance_table: numpy.ndarray,
    neighbor_count: int = 6,
    distance_ratio: float = 1.25,
) -> list:
    """
    每個圓找最近的 neighbor_count 個圓當鄰居，回傳 [[i, j, 距離], ...]。
        distance_table  get_distance_table 的回傳值
        neighbor_count  每個圓找幾個鄰居：方格排列通常是 4，蜂巢排列通常是 6
        distance_ratio  距離過濾：超過「所有配對距離的中位數 × distance_ratio」
                        的配對刪掉（邊緣的圓湊不滿鄰居時，會抓到比較遠的圓）
    每一對鄰居只出現一次，而且 i < j。
    """

    circle_count = len(distance_table)
    pair_set = set()
    for circle_index in range(circle_count):
        # 由近到遠排序，第 0 個是自己（距離 0），所以從第 1 個開始取
        nearest_indices = numpy.argsort(distance_table[circle_index])
        for neighbor_index in nearest_indices[1:neighbor_count + 1]:
            # 小的編號放前面，同一對鄰居不管從哪邊找到，都只會存一次
            smaller_index = int(min(circle_index, neighbor_index))
            larger_index = int(max(circle_index, neighbor_index))
            pair_set.add((smaller_index, larger_index))

    candidate_pairs = []
    for smaller_index, larger_index in sorted(pair_set):
        pair_distance = float(distance_table[smaller_index, larger_index])
        candidate_pairs.append([smaller_index, larger_index, pair_distance])
    # 一對都沒有（例如只有一個圓）時，不用過濾
    if len(candidate_pairs) == 0:
        return candidate_pairs

    # 距離過濾：真正的鄰居佔大多數，中位數會落在真正的鄰居距離附近
    median_distance = numpy.median(
        [candidate_pair[2] for candidate_pair in candidate_pairs]
    )
    distance_limit = median_distance * distance_ratio
    neighbor_pairs = []
    for candidate_pair in candidate_pairs:
        if candidate_pair[2] > distance_limit:
            continue
        neighbor_pairs.append(candidate_pair)
    return neighbor_pairs

def get_distance_summary(
    neighbor_pairs: list
) -> dict:
    """
    鄰居距離的統計，回傳 {'count', 'mean', 'median', 'std', 'min', 'max'}。
        neighbor_pairs  get_neighbor_pairs 的回傳值
    沒有任何鄰居時，數量是 0，其他都是 None。
    """

    pair_distances = numpy.array(
        [neighbor_pair[2] for neighbor_pair in neighbor_pairs]
    )
    if len(pair_distances) == 0:
        distance_summary = {
            'count': 0, 'mean': None, 'median': None,
            'std': None, 'min': None, 'max': None,
        }
        return distance_summary

    distance_summary = {
        'count': len(pair_distances),
        'mean': float(pair_distances.mean()),
        'median': float(numpy.median(pair_distances)),
        'std': float(pair_distances.std()),
        'min': float(pair_distances.min()),
        'max': float(pair_distances.max()),
    }
    return distance_summary
