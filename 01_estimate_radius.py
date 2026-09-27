import cv2
import os
import pathlib
import numpy

class Radius:

    def __init__(self, checkpoint_path: str = '.tmp/') -> None:
        """
        checkpoint_path: 中間處理的暫存檔案保存位置。
        """

        self.checkpoint_path = checkpoint_path
        return

    def read_gray_image(self, file_path: str) -> bool:
        """讀取圖片檔案"""

        self.gray_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # 計算用
        self.color_image = cv2.imread(file_path)                        # 畫圖用
        return True

    def get_scale_ratio(self, size_bound: int) -> float:
        """獲得縮小比例"""

        scale_ratio = size_bound / max(self.gray_image.shape[:2])
        return scale_ratio

    def get_radius_ratio(
        self, estimated_radius: float, target_radius: float = 40
    ) -> float:
        """
        獲得縮放比例：讓半徑 estimated_radius 的圓，縮放後變成 target_radius。
        第二階段用。第一階段用 get_scale_ratio（依圖的大小）。

        例如 r = 118.9：40 ÷ 118.9 = 0.336，圖縮小成 0.336 倍
             r = 20：   40 ÷ 20    = 2.0，  圖放大成 2 倍

        所有圖的圓都變成半徑 40 左右，後面的參數（Hough 找 24～56、
        模糊 2.0）才能每張圖通用，不用每張圖各自調整。

        參數：
            estimated_radius  第一階段估計出的 r（原圖像素）
            target_radius     縮放後希望圓的半徑是多少（notebook 的 R_STD）
        """

        scale_ratio = target_radius / estimated_radius
        return scale_ratio

    def get_gray_image(self, scale_ratio: float = 1.0) -> numpy.ndarray:
        """
        獲得灰階的圖片陣列，並依照縮放比例縮放。
            scale_ratio < 1：縮小（第一階段、圓比較大的圖）
            scale_ratio = 1：不縮放（第三階段，在原圖上修準）
            scale_ratio > 1：放大（第二階段、圓比較小的圖，例如 r = 20）
        """

        # gray_image = cv2.cvtColor(self.gray_image, cv2.COLOR_BGR2GRAY)
        if scale_ratio == 1.0:

            scale_image = self.gray_image.copy()
            return scale_image

        # 縮小和放大用不同的方法，畫質比較好：
        #   INTER_AREA：縮小時把好幾個像素平均成一個，不會有鋸齒或雜點
        #   INTER_CUBIC：放大時用周圍 4×4 個像素算出新像素，邊緣比較平滑
        if scale_ratio < 1.0:
            resize_method = cv2.INTER_AREA
        else:
            resize_method = cv2.INTER_CUBIC

        scale_image = cv2.resize(
            self.gray_image, None,
            fx=scale_ratio, fy=scale_ratio,
            interpolation=resize_method,
        )
        return scale_image

    def get_gray_edges(
        self, gray_image: numpy.ndarray, blur_sigma: float = 1.5
    ) -> tuple:
        """
        算出每個像素的邊緣資訊，回傳 (gradient, magnitude)：
            gradient   (g_x, g_y) 兩張陣列，合起來是指向變亮方向的箭頭
                g_x     往右走，亮度變化多少（變亮 = 正，變暗 = 負）
                g_y     往下走，亮度變化多少
            magnitude  箭頭長度，也就是邊緣強度 √(g_x² + g_y²)
        每張陣列都跟 gray_image 一樣大。
        """

        blurred_image = cv2.GaussianBlur(gray_image, (0, 0), blur_sigma)
        gray_gradient = (
            cv2.Sobel(blurred_image, cv2.CV_32F, 1, 0),
            cv2.Sobel(blurred_image, cv2.CV_32F, 0, 1)
        )
        gray_magnitude = cv2.magnitude(*gray_gradient)
        return gray_gradient, gray_magnitude

    def save_image(
        self,
        target_image: numpy.ndarray,
        file_name: str,
        circle_centers: list = None,
        circle_radii: list = None,
        circle_color: tuple | list = (0, 255, 0),
        line_thickness: int = 1,
    ) -> str:
        """
        把圖片存到 checkpoint_path 底下，回傳存檔路徑。

        參數：
            target_image     要存的圖片（灰階或彩色都可以）
            file_name        檔名，例如 'candidates.png'
            circle_centers   要畫的圓心 [[x, y], ...]。
                             不傳（None）就只存圖，什麼都不畫。
            circle_radii     每個圓心對應的半徑 [r, ...]，順序要跟圓心一樣。
                             不傳就只畫圓心的點；有傳就再畫出圓周。
            circle_color     顏色，順序是 (B, G, R)，預設綠色。
                             傳一個顏色 → 所有圓都用它；
                             傳一串顏色 [(B, G, R), ...] → 每個圓各用一個
            line_thickness   圓周的線條粗細（像素）
        """

        if circle_centers is not None:
            if circle_radii is not None and \
                    len(circle_radii) != len(circle_centers):
                raise ValueError('circle_radii 跟 circle_centers 數量不一樣')

            # 灰階圖每格只有一個亮度值，畫不出顏色，先轉成 BGR 彩色
            # 轉換或 copy 都會產生新的圖，不會改到原本的 target_image
            if target_image.ndim == 2:
                target_image = cv2.cvtColor(target_image, cv2.COLOR_GRAY2BGR)
            else:
                target_image = target_image.copy()

            # 只傳一個顏色 (B, G, R)：每個圓都用同一個，複製成一串
            if isinstance(circle_color, tuple):
                circle_color = [circle_color] * len(circle_centers)

            for index, (c_x, c_y) in enumerate(circle_centers):
                # cv2.circle 只接受整數座標
                c_center = (int(round(c_x)), int(round(c_y)))
                # 圓心：畫一個實心小點（線條粗細 -1 = 塗滿）
                cv2.circle(
                    target_image, c_center, line_thickness + 1,
                    circle_color[index], -1,
                )
                # 有半徑才畫圓周
                if circle_radii is not None:
                    c_r = int(round(circle_radii[index]))
                    cv2.circle(
                        target_image, c_center, c_r,
                        circle_color[index], line_thickness,
                    )

        os.makedirs(self.checkpoint_path, exist_ok=True)
        file_path = os.path.join(self.checkpoint_path, file_name)
        cv2.imwrite(file_path, target_image)
        return file_path

    def get_circle_candidates(
        self,
        gray_image: numpy.ndarray,
        blur_sigma: float = 1.5,
        center_distance: float = 4,
        canny_threshold: float = 80,
        vote_threshold: float = 25,
        minimum_radius: int = 6,
        maximum_radius: int = None,
    ) -> list:
        """
        猜出可能的圓，回傳 [[x, y, r], ...]（縮小後的座標）。
        條件故意設得寬鬆，寧可多猜，假圓交給後面的打分數淘汰。
        一個都沒找到時回傳空的 list。

        用法：預設值是第一階段（估計 r）用的
            第一階段：還不知道 r，什麼大小都找
                get_circle_candidates(gray_image)
                → 半徑 6 ～ 圖的一半
            第二階段：已經知道 r，圖縮放到半徑 = 40，只找 40 附近的圓
                get_circle_candidates(
                    gray_image, blur_sigma=2.0, center_distance=10,
                    vote_threshold=12, minimum_radius=24, maximum_radius=56,
                )
                → 內部的小圓一開始就不會被猜出來

        參數：
            gray_image       灰階圖（已經縮放好的）
            blur_sigma       先模糊多少再找圓
            center_distance  兩個圓心至少隔幾個像素（很小 → 候選很多）
            canny_threshold  Hough 內部找邊緣用的門檻
            vote_threshold   至少幾票才算候選（偏低 → 候選很多）
            minimum_radius   最小半徑（縮放後的像素）
            maximum_radius   最大半徑（縮放後的像素）。
                             None = 圖的短邊的一半
        """

        if maximum_radius is None:
            maximum_radius = min(gray_image.shape) // 2

        blurred_image = cv2.GaussianBlur(gray_image, (0, 0), blur_sigma)
        candidate_results = cv2.HoughCircles(
            blurred_image, cv2.HOUGH_GRADIENT, dp=1,
            minDist=center_distance,
            param1=canny_threshold,
            param2=vote_threshold,
            minRadius=minimum_radius,
            maxRadius=maximum_radius,
        )
        if candidate_results is None:
            circle_candidates = []
            return circle_candidates

        circle_candidates = numpy.array(candidate_results[0]).tolist()
        return circle_candidates

    def get_circle_scores(
        self,
        circle_candidates: list,
        gray_edges: tuple,
        strong_quantile: float = 0.8,
        point_count: int = 72,
        buffer_width: int = 2,
        buffer_ratio: float = 0.0,
        direction_cosine: float = 0.8,
    ) -> list:
        """
        替每個候選圓打分數，分辨真圓和假圓。回傳每個圓的分數 [0.9, 0.1, ...]。

        打分數的方法：沿圓周放一圈檢查點，每個點問兩個問題
            一、這裡有強邊緣嗎？
            二、邊緣的箭頭有沒有沿著半徑方向？（指向或背向圓心都算）
        兩題都「是」才得一分。分數 = 得分的點 ÷ 檢查點總數（0～1）。

        用法：同一個方法，快篩和正式打分數只差在參數
            快篩（refine 之前，先丟掉明顯的假圓）：
                get_circle_scores(circles, gray_edges,
                                  point_count=36, buffer_ratio=0.06)
                → 點比較少（快），看得比較寬（圓心還沒修正，可能偏了）
                → 之後用 get_filtered_circles(..., minimum_score=0.55) 過濾
            正式（refine 之後，確定是不是真圓）：
                get_circle_scores(circles, gray_edges)
                → 用預設值：72 個點，固定往內外看 2 格
                → 之後用 get_filtered_circles(..., minimum_score=0.7) 過濾

        參數：
            circle_candidates  要打分數的圓 [[x, y, r], ...]
            gray_edges         get_gray_edges 的回傳值 (gradient, magnitude)
            strong_quantile    magnitude 排在前多少才算「強邊緣」。
                               0.8 = 比 80% 的像素強，也就是最強的前 20%。
                               用比例而不用固定數字，是因為每張圖的亮度不同。
            point_count        圓周上放幾個檢查點。72 = 每 5 度一個點。
                               點越多越精確但越慢，快速篩選時可以用 36。
            buffer_width       每個檢查點往內、往外各多看幾格像素，
                               只要其中一格符合就得分。
                               Hough 猜的半徑常差 1～2 像素，這是容許的誤差。
            buffer_ratio       依照半徑再加寬，實際看的格數是
                               max(buffer_width, buffer_ratio × r)。
                               0.0：不加寬，每個圓都看 buffer_width 格（正式）
                               0.06：半徑 100 看 6 格、半徑 20 看 2 格（快篩）
                               圓心還沒修正時大圓偏得多，所以要看寬一點。
            direction_cosine   箭頭跟半徑方向夾角的 cos 要大於多少。
                               1.0 = 完全沿著半徑，0 = 完全垂直。
                               0.8 ≈ 夾角 37 度以內。
                               圓內部的紋路也是強邊緣，但箭頭方向亂七八糟，
                               靠這個條件把它們排除。
        """

        gray_gradient, gray_magnitude = gray_edges
        # g_x, g_y：箭頭（gradient）的 x、y 分量，從圖片算出來的
        g_x, g_y = gray_gradient

        # 門檻在這裡只算一次，所有圓共用（每個圓都算會很慢）
        strong_threshold = float(
            numpy.quantile(gray_magnitude, strong_quantile)
        )

        # ---- 第 1 塊：檢查點的角度（每個圓都一樣）----
        # 72 個角度：0, 5, 10, ... 355 度（用弧度表示，2π = 360 度）
        # endpoint=False：不要 360 度，因為 360 度跟 0 度是同一個點
        point_angles = numpy.linspace(
            0, 2 * numpy.pi, 
            point_count,
            endpoint=False
        )
        # ux, uy：單位向量（unit vector），從圓心往每個角度走一步的方向
        # 就是每個角度的 cos、sin。純幾何，跟圖片無關（對照 gx, gy）
        # [:, None] 把形狀從 (72,) 變成 (72, 1)，後面才能跟 b_w 配對
        u_x = numpy.cos(point_angles)[:, None]
        u_y = numpy.sin(point_angles)[:, None]

        circle_scores = []
        for c_x, c_y, c_r in circle_candidates:
            # center_x, center_y, circle_radius = circle_candidate

            # ---- 往內、往外各看幾格（每個圓可能不一樣，所以放在迴圈裡）----
            # c_b：這個圓要看幾格（b = buffer）
            # buffer_ratio = 0 時，就是 buffer_width，每個圓都一樣
            c_b = max(buffer_width, int(buffer_ratio * c_r))
            # c_b=2 → [-2, -1, 0, 1, 2]
            # [None, :] 把形狀從 (5,) 變成 (1, 5)
            b_w = numpy.arange(-c_b, c_b + 1)[None, :]

            # ---- 第 2 塊：算出 72 × 5 個檢查點在圖上的座標 ----
            # 每一欄離圓心多遠：[r-2, r-1, r, r+1, r+2]，形狀 (1, 5)
            p_r = c_r + b_w
            # p_x, p_y：檢查點的座標（p = point）
            # 座標 = 圓心 + 方向 × 距離
            # (72, 1) × (1, 5) → 自動展開成 (72, 5) 的表格
            # 像素的位置只能是整數，所以四捨五入後轉成 int
            p_x = numpy.round(c_x + u_x * p_r).astype(int)
            p_y = numpy.round(c_y + u_y * p_r).astype(int)

            # ---- 第 3 塊：處理跑到圖片外面的檢查點 ----
            # i_h, i_w：圖片的高、寬（i = image）
            i_h, i_w = gray_magnitude.shape
            # 每一格是否在圖片裡面 → (72, 5) 的 True / False 表格
            # .all(axis=1)：一個角度的 5 格「全部」在圖內，這個角度才算數
            # 結果 p_in 形狀是 (72,)，每個角度一個 True / False
            p_in = (
                (p_x >= 0) & (p_x < i_w) & (p_y >= 0) & (p_y < i_h)
            ).all(axis=1)
            # 在圖內的角度不到一半 → 這個圓大部分在圖外，不可信，給 0 分
            if p_in.sum() < point_count * 0.5:
                circle_scores.append(0.0)
                continue
            # 把圖外的座標拉回圖片邊界，只是為了下一步取值時不會報錯
            # 這些點在最後算分時會被 p_in 排除，不加分也不扣分
            p_x = numpy.clip(p_x, 0, i_w - 1)
            p_y = numpy.clip(p_y, 0, i_h - 1)

            # ---- 第 4 塊：問兩個問題，算出分數 ----
            # 取出每個檢查點的 magnitude → (72, 5)
            # 注意順序是 [y, x]：陣列先選「第幾列」，再選「第幾欄」
            p_m = gray_magnitude[p_y, p_x]
            # 問題一：這裡有強邊緣嗎？ → (72, 5) 的 True / False
            p_strong = p_m > strong_threshold
            # 問題二：箭頭有沒有沿著半徑方向？
            # 箭頭 (g_x, g_y) 跟方向 (u_x, u_y) 做內積，再除以箭頭長度
            # 得到兩者夾角的 cos：1 = 同方向，0 = 垂直，-1 = 反方向
            # 取絕對值：指向圓心（-1）或背向圓心（1）都算
            # + 1e-6：避免箭頭長度是 0 時變成除以 0
            p_cos = numpy.abs(
                g_x[p_y, p_x] * u_x + g_y[p_y, p_x] * u_y
            ) / (p_m + 1e-6)
            p_radial = p_cos > direction_cosine
            # 兩題都「是」才得分
            # .any(axis=1)：一個角度的 5 格裡「任一格」得分，這個角度就得分
            # 結果 p_hit 形狀是 (72,)
            p_hit = (p_strong & p_radial).any(axis=1)
            # 分數 = 圖內得分的角度 ÷ 圖內的角度
            c_s = (p_hit & p_in).sum() / p_in.sum()
            circle_scores.append(float(c_s))
            continue
        
        return circle_scores

    def get_filtered_circles(
        self,
        circle_candidates: list,
        circle_scores: list,
        minimum_score: float = 0.7,
    ) -> list:
        """
        過濾：只留下分數夠高的圓，並把圓和分數綁在一起。
        回傳 [[x, y, r, score], ...]。

        參數：
            circle_candidates  圓 [[x, y, r], ...]
            circle_scores      get_circle_scores 的回傳值，順序跟圓一樣
            minimum_score      分數至少多少才留下（notebook 的 MIN_SCORE）。
                               正式打分數用 0.7；快篩時放寬成 0.55。
        """

        # 數量不一樣代表哪裡傳錯了，zip 會默默丟掉多出來的，所以先檢查
        if len(circle_candidates) != len(circle_scores):
            raise ValueError('circle_candidates 跟 circle_scores 數量不一樣')

        filtered_circles = []
        # zip：把圓和分數一對一配好，一次拿一組
        for (c_x, c_y, c_r), c_s in zip(circle_candidates, circle_scores):
            if c_s < minimum_score:
                continue
            filtered_circles.append([c_x, c_y, c_r, c_s])
            continue

        return filtered_circles

    def get_refined_circles(
        self,
        circle_candidates: list,
        gray_magnitude: numpy.ndarray,
        ray_count: int = 90,
        ring_ratio: float = 0.15,
        change_ratio: float = 0.2,
        keep_score: bool = False,
    ) -> list:
        """
        微調圓心和半徑，回傳修正後的圓 [[x, y, r], ...]，順序跟輸入一樣。
        Hough 給的圓心常常偏一點，打分數前先修正，免得真圓被冤枉扣分。

        用法：
            第一、二階段：用預設值，回傳 [[x, y, r], ...]
                → 接著會重新打分數，舊分數不需要
            第三階段：keep_score=True，回傳 [[x, y, r, score], ...]
                → 保留第二階段的分數（這個分數是 refine 之前算的）

        作法（每個圓）：
            1. 從圓心射出 ray_count 條射線，每條只在 r 附近的環帶裡
               找 magnitude 最強的那一格 → 得到一圈「真正的邊緣點」
            2. 用這些點擬合出一個最貼合的圓（最小平方法）
        修正不可靠時（邊緣點太少、改變太多），就維持原本的圓。

        參數：
            circle_candidates  要微調的圓 [[x, y, r], ...]。
                               也可以傳 [[x, y, r, score], ...]，只會用前三個。
            gray_magnitude     get_gray_edges 回傳的 magnitude（邊緣強度）
            ray_count          射線數量。90 = 每 4 度一條。
            ring_ratio         環帶寬度：只在 (1 - 0.15)r ～ (1 + 0.15)r 裡找。
                               太寬會找到隔壁圓的邊緣，太窄會錯過偏掉的邊緣。
            change_ratio       修正幅度上限：圓心移動或半徑改變超過 0.2r，
                               代表可能被別的邊緣拉走了，不採用。
            keep_score         True = 把輸入的分數（第 4 個數字）接在輸出後面。
                               輸入的圓沒有分數時會報錯。
        """

        # ---- 第 1 塊：射線的方向、每條射線上的取樣距離（比例）----
        # 90 個角度，跟 get_circle_scores 的第 1 塊一樣
        ray_angles = numpy.linspace(
            0, 2 * numpy.pi,
            ray_count,
            endpoint=False
        )
        # u_x, u_y：單位向量，每條射線的方向，形狀 (90, 1)
        u_x = numpy.cos(ray_angles)[:, None]
        u_y = numpy.sin(ray_angles)[:, None]
        # s_r：每條射線上取 31 個點，離圓心的距離是 r 的幾倍（s = sample）
        # 0.85, 0.86, ... 1.15，形狀 (1, 31)
        # 乘上半徑之後，就是真正的距離
        s_r = numpy.linspace(1 - ring_ratio, 1 + ring_ratio, 31)[None, :]
        # i_h, i_w：圖片的高、寬（i = image），每個圓都一樣，放迴圈外
        i_h, i_w = gray_magnitude.shape
        # r_i：0, 1, 2, ... 89，每條射線的編號（r = ray），第 2 塊取值時用
        r_i = numpy.arange(ray_count)

        refined_circles = []
        for circle_candidate in circle_candidates:
            # 只用前三個數字，有沒有 score 都可以
            c_x, c_y, c_r = circle_candidate[:3]
            # c_k：要接回去的分數（k = keep）。不保留時是空的 []
            c_k = []
            if keep_score:
                if len(circle_candidate) < 4:
                    raise ValueError('keep_score=True，但輸入的圓沒有分數')
                c_k = [circle_candidate[3]]

            # ---- 第 2 塊：沿著每條射線，找出最強的邊緣點 ----
            # s_x, s_y：每條射線上 31 個取樣點的座標，形狀 (90, 31)
            # 距離 = 半徑 × 比例（0.85r ～ 1.15r）
            s_x = numpy.round(c_x + u_x * (c_r * s_r)).astype(int)
            s_y = numpy.round(c_y + u_y * (c_r * s_r)).astype(int)
            # s_in：這條射線的 31 個點「全部」在圖內，才採用這條射線
            # 形狀 (90,)，跟打分數第 3 塊的 p_in 一樣的寫法
            s_in = (
                (s_x >= 0) & (s_x < i_w) & (s_y >= 0) & (s_y < i_h)
            ).all(axis=1)
            # 先拉回圖內，只是為了下一行取值不報錯；圖外的射線最後會丟掉
            s_x = numpy.clip(s_x, 0, i_w - 1)
            s_y = numpy.clip(s_y, 0, i_h - 1)
            # s_m：每個取樣點的 magnitude，形狀 (90, 31)
            s_m = gray_magnitude[s_y, s_x]
            # s_j：每條射線上「最強的是第幾格」，形狀 (90,)
            s_j = numpy.argmax(s_m, axis=1)
            # e_x, e_y, e_m：每條射線找到的邊緣點座標和強度（e = edge）
            # [r_i, s_j]：第 0 條射線取第 s_j[0] 格、第 1 條取第 s_j[1] 格…
            e_x = s_x[r_i, s_j][s_in]
            e_y = s_y[r_i, s_j][s_in]
            e_m = s_m[r_i, s_j][s_in]
            # 最後的 [s_in]：只留下全部在圖內的射線

            # ---- 第 3 塊：用這圈邊緣點擬合出一個圓（最小平方法）----
            # 在圖內的射線不到一半 → 邊緣點太少，擬合不可靠，維持原樣
            if len(e_m) < ray_count * 0.5:
                refined_circles.append([c_x, c_y, c_r] + c_k)
                continue
            # 丟掉最弱的 25% 邊緣點：那些射線可能沒碰到真的邊緣，
            # 只是在雜訊裡挑了「相對最強」的一格
            e_keep = e_m > numpy.percentile(e_m, 25)
            e_x = e_x[e_keep].astype(float)
            e_y = e_y[e_keep].astype(float)
            # 圓的方程式 (x - a)² + (y - b)² = R² 展開整理後變成
            #     2a·x + 2b·y + c = x² + y²      其中 c = R² - a² - b²
            # 對 a, b, c 來說這是「一次方程式」，每個邊緣點給一條：
            #     [2x, 2y, 1] · [a, b, c] = x² + y²
            # 幾十個點 → 幾十條方程式，但只有 3 個未知數，不可能全部剛好成立
            # lstsq 找出「讓所有方程式整體誤差最小」的 a, b, c
            f_A = numpy.column_stack([2 * e_x, 2 * e_y, numpy.ones(len(e_x))])
            f_b = e_x ** 2 + e_y ** 2
            f_x, f_y, f_c = numpy.linalg.lstsq(f_A, f_b, rcond=None)[0]
            # f_x, f_y：擬合出來的圓心 (a, b)（f = fit）
            # f_r：半徑。由 c = R² - a² - b² 反推 R = √(c + a² + b²)
            f_r = numpy.sqrt(f_c + f_x ** 2 + f_y ** 2)

            # ---- 第 4 塊：檢查修正幅度，太大就不採用 ----
            # 圓心移動的距離 √(Δx² + Δy²)
            f_shift = numpy.hypot(f_x - c_x, f_y - c_y)
            # 圓心移太多、或半徑改太多（超過 0.2r）→ 可能被隔壁圓的邊緣
            # 拉走了，寧可維持原樣
            # 另外，點排得很怪時半徑可能算不出來（√ 負數 = nan），也維持原樣
            if (
                f_shift > change_ratio * c_r
                or abs(f_r - c_r) > change_ratio * c_r
                or not numpy.isfinite(f_r)
            ):
                refined_circles.append([c_x, c_y, c_r] + c_k)
                continue

            # 通過檢查，採用修正後的圓
            refined_circles.append([float(f_x), float(f_y), float(f_r)] + c_k)
            continue

        return refined_circles

    def get_unique_circles(
        self,
        filtered_circles: list,
        overlap_ratio: float = 0.75,
        inside_ratio: float = 0.85,
        scale_ratio: float = 1.0,
    ) -> list:
        """
        NMS（Non-Maximum Suppression，非極大值抑制）：
        擠在一起的圓只留最好的一個，回傳不重複的圓 [[x, y, r, score], ...]。
        順便可以把座標換回原圖大小（見 scale_ratio）。

        作法像排隊選人：
            1. 依「分數 × √半徑」由高到低排隊
            2. 一個一個看，跟「已經選上的圓」衝突就淘汰，不衝突就選上

        用法：
            每一層（600、300、150）：傳入那一層的縮放比例，
                get_unique_circles(filtered_circles, scale_ratio=scale_ratio)
                → 回傳的圓已經換回原圖座標，三層才能放在一起比
            三層合併之後：圓已經是原圖座標，不用傳，
                get_unique_circles(merged_circles)
                → scale_ratio 預設 1.0，不換算

        參數：
            filtered_circles  get_filtered_circles 的回傳值
                            [[x, y, r, score], ...]
            overlap_ratio   重疊太多：圓心距離 < overlap_ratio × (r1 + r2)
                            兩個圓剛好貼在一起時，距離 = r1 + r2，不算衝突；
                            0.75 代表圓心要靠得比這更近才算重疊太多。
            inside_ratio    包在裡面：圓心距離 < inside_ratio × 較大的半徑
                            小圓的圓心深入大圓內部就淘汰（例如細胞裡的小團塊）。
            scale_ratio     輸入的圓是在縮小幾倍的圖上找到的（get_scale_ratio）
                            回傳前 x, y, r 都會除以它，換回原圖座標；分數不變。
                            1.0 = 輸入已經是原圖座標，不換算。
                            兩個門檻都是比例，所以先換算或後換算，
                            選出來的圓都一樣。
        """

        # ---- 第 1 塊：排隊 ----
        # 排序的依據：分數 × √半徑，大的排前面（所以加負號）
        # 乘上 √半徑：分數差不多時，讓大圓先選上，
        # 大圓裡面的小圓之後就會因為「包在裡面」被淘汰
        sorted_circles = sorted(
            filtered_circles,
            key=lambda c: -(c[3] * numpy.sqrt(c[2])),
        )

        # ---- 第 2 塊：一個一個看，跟已經選上的圓比 ----
        unique_circles = []
        for c_x, c_y, c_r, c_s in sorted_circles:
            is_conflict = False
            # k_x, k_y, k_r：已經選上（kept）的圓
            for k_x, k_y, k_r, _ in unique_circles:
                # 兩個圓心的距離 √(Δx² + Δy²)
                c_d = numpy.hypot(c_x - k_x, c_y - k_y)
                # 衝突一：重疊太多
                if c_d < overlap_ratio * (c_r + k_r):
                    is_conflict = True
                    break
                # 衝突二：包在裡面
                if c_d < inside_ratio * max(c_r, k_r):
                    is_conflict = True
                    break
            # 跟所有已選上的圓都不衝突，才選上
            if not is_conflict:
                unique_circles.append([c_x, c_y, c_r, c_s])
            continue

        # ---- 第 3 塊：換回原圖座標 ----
        # 例如 L600 的 scale_ratio = 0.462：r = 57.6 → 57.6 ÷ 0.462 = 124.8
        # 分數是比例（0～1），跟圖的大小無關，不用換
        unique_circles = [
            [c_x / scale_ratio, c_y / scale_ratio, c_r / scale_ratio, c_s]
            for c_x, c_y, c_r, c_s in unique_circles
        ]

        return unique_circles

    def get_partial_circles(self, target_circles: list) -> list:
        """
        判斷每個圓有沒有被圖片邊界切到（partial），第三階段用。
        輸入 [[x, y, r, ...], ...]，回傳時每個圓的最後面多一個 True / False。
        例如 [x, y, r, score] → [x, y, r, score, partial]

        被切到的圓，圓心可能比較不準，畫圖時可以用別的顏色標出來。
        圓要是原圖座標，因為是跟原圖的邊界比。
        """

        # i_h, i_w：原圖的高、寬
        i_h, i_w = self.gray_image.shape

        partial_circles = []
        for target_circle in target_circles:
            c_x, c_y, c_r = target_circle[:3]
            # 圓心到上、下、左、右四個邊界的距離，最短的那個
            # 比半徑還短 → 圓一定有一部分跑到圖外
            c_edge = min(c_x, c_y, i_w - 1 - c_x, i_h - 1 - c_y)
            is_partial = bool(c_edge < c_r)
            partial_circles.append(list(target_circle) + [is_partial])

        return partial_circles

    def get_estimated_radius(
        self,
        unique_circles: list,
        hill_width: float = 0.15,
    ) -> float:
        """
        從一堆圓估計出 r：「哪種大小的圓，佔了這張圖最多的面積？」
        回傳 r（原圖像素）。只用到半徑，圓心和分數不會用到。

        作法：
            每個圓在自己的半徑位置放一座小山，山的高度 = 面積 r²，
            全部疊起來，最高點的位置就是 r。
            內部小圓、雜訊再多，面積都很小，搶不走答案。

        參數：
            unique_circles  get_unique_circles 的回傳值 [[x, y, r, score], ...]
                            注意：要先換回原圖座標！
                            （三層合併、再做一次 NMS 之後的圓）
            hill_width      每座小山的寬度，在 log 尺度上。
                            0.15 ≈ 半徑 ±15% 的範圍會互相疊加，
                            所以 100 和 110 會被當成「差不多大」。
        """

        # ---- 第 1 塊：換到 log 尺度 ----
        # 為什麼用 log：讓「比例」相同的差距看起來一樣寬
        # 例如 10 → 11 和 100 → 110 都是大 10%，在 log 上距離一樣
        # 這樣同一座山的寬度，不管圓大圓小，代表的都是 ±15%
        # c_r：從每個圓 [x, y, r, score] 取出第 2 個，也就是半徑
        c_r = numpy.array([c[2] for c in unique_circles], dtype=float)
        l_r = numpy.log(c_r)

        # ---- 第 2 塊：準備橫軸 ----
        # g_r：600 個候選位置，從最小的半徑再往左一點，到最大的再往右一點
        # （g = grid），形狀 (600, 1)
        # 左右多留 0.3，山的兩側才不會被切掉
        g_r = numpy.linspace(l_r.min() - 0.3, l_r.max() + 0.3, 600)[:, None]

        # ---- 第 3 塊：每個圓放一座小山 ----
        # 高斯函數 exp(-½ × (距離 / 寬度)²)：
        # 剛好在圓的半徑上 = 1，離得越遠越接近 0
        # (600, 1) 跟 (1, N) → (600, N) 的表格：每個位置 × 每個圓
        h_s = numpy.exp(-0.5 * ((g_r - l_r[None, :]) / hill_width) ** 2)

        # ---- 第 4 塊：山的高度乘上面積，全部疊起來，找最高點 ----
        # h_s × r²：大圓的山比較高（面積大，票數多）
        # .sum(axis=1)：每個位置把所有圓的山加起來，形狀 (600,)
        a_s = (h_s * c_r[None, :] ** 2).sum(axis=1)
        # argmax：最高點在第幾個位置；再從 log 換回一般的半徑
        estimated_radius = float(numpy.exp(g_r[numpy.argmax(a_s), 0]))

        return estimated_radius

    pass

if __name__=='__main__':

    # 流程圖和每一步的說明：pipeline_flow.md
    file_path = './sample_01.jpg'
    radius = Radius()
    radius.read_gray_image(file_path)
    print('原圖大小：', radius.gray_image.shape)

    # ================================================================
    # 第一階段：估計 r（對應 01_estimate_radius.ipynb）
    # 還不知道圓多大，所以把圖縮成 600、300、150 三種大小，每層都找一次，
    # 不管圓多大，總有一層剛好適合它。
    # ================================================================
    merged_circles = []
    for size_bound in (600, 300, 150):
        scale_ratio = radius.get_scale_ratio(size_bound)
        # 圖本身比這層還小，放大沒有意義，跳過
        if scale_ratio > 1:
            continue
        gray_image = radius.get_gray_image(scale_ratio)
        gray_edges = radius.get_gray_edges(gray_image)

        # Hough 猜圓：用預設值，什麼大小都找，寧可多猜
        circle_candidates = radius.get_circle_candidates(gray_image)

        # 快篩：36 個點、看寬一點，門檻 0.55，先丟掉明顯的假圓
        circle_scores = radius.get_circle_scores(
            circle_candidates, gray_edges,
            point_count=36, buffer_ratio=0.06,
        )
        filtered_circles = radius.get_filtered_circles(
            circle_candidates, circle_scores, minimum_score=0.55,
        )

        # refine：把圓心挪準，分數才不會被低估
        refined_circles = radius.get_refined_circles(
            filtered_circles, gray_edges[1],
        )

        # 正式打分數：72 個點，門檻 0.7
        circle_scores = radius.get_circle_scores(refined_circles, gray_edges)
        filtered_circles = radius.get_filtered_circles(
            refined_circles, circle_scores,
        )

        # NMS：同一顆圓只留一個，並換回原圖座標，三層才能放在一起
        unique_circles = radius.get_unique_circles(
            filtered_circles, scale_ratio=scale_ratio,
        )
        print(f'  L{size_bound}：{len(unique_circles)} 個圓')
        merged_circles += unique_circles

    # 同一顆圓可能在好幾層都被找到，合併後再 NMS 一次
    unique_circles = radius.get_unique_circles(merged_circles)
    # 圓太少，半徑分布沒有意義（notebook 的 MIN_CIRCLES）
    if len(unique_circles) < 3:
        raise SystemExit(f'只找到 {len(unique_circles)} 個圓，估不出 r')
    # 面積加權的半徑分布，最高點就是 r
    estimated_radius = radius.get_estimated_radius(unique_circles)
    print(f'第一階段：合併後 {len(unique_circles)} 個圓，'
          f'r = {estimated_radius:.1f}')

    # ================================================================
    # 第二階段：找出一批圓（對應 02_detect_circles.ipynb）
    # 已經知道 r，把圖縮放到「圓的半徑 = 40」，
    # 後面的參數（模糊 2.0、Hough 找 24～56）每張圖都能通用。
    # ================================================================
    scale_ratio = radius.get_radius_ratio(estimated_radius)
    gray_image = radius.get_gray_image(scale_ratio)
    gray_edges = radius.get_gray_edges(gray_image, blur_sigma=2.0)

    # Hough 猜圓：只找半徑 40 附近（24～56），內部小圓一開始就不會被猜出來
    circle_candidates = radius.get_circle_candidates(
        gray_image, blur_sigma=2.0, center_distance=10,
        vote_threshold=12, minimum_radius=24, maximum_radius=56,
    )

    # 快篩：強邊緣門檻改成 65%，分數門檻 0.6 - 0.15 = 0.45
    circle_scores = radius.get_circle_scores(
        circle_candidates, gray_edges, strong_quantile=0.65,
        point_count=36, buffer_ratio=0.06,
    )
    filtered_circles = radius.get_filtered_circles(
        circle_candidates, circle_scores, minimum_score=0.45,
    )

    # refine
    refined_circles = radius.get_refined_circles(
        filtered_circles, gray_edges[1],
    )

    # 正式打分數：門檻 0.6
    circle_scores = radius.get_circle_scores(
        refined_circles, gray_edges, strong_quantile=0.65,
    )
    filtered_circles = radius.get_filtered_circles(
        refined_circles, circle_scores, minimum_score=0.6,
    )

    # NMS：重疊門檻改成 0.6，並換回原圖座標
    detected_circles = radius.get_unique_circles(
        filtered_circles, overlap_ratio=0.6, scale_ratio=scale_ratio,
    )
    print(f'第二階段：找到 {len(detected_circles)} 個圓')

    # ================================================================
    # 第三階段：在原圖上修準（對應 03_refine_circles.ipynb）
    # 第二階段是在縮小的圖上找的，1 個像素 = 原圖好幾個像素，
    # 回到原圖再 refine 一次，圓心更準。
    # ================================================================
    gray_image = radius.get_gray_image()          # 不縮放 = 原圖
    # 模糊 = 半徑的 1/20，跟第二階段一樣的比例，換算回原圖大小
    median_radius = float(numpy.median([c[2] for c in detected_circles]))
    gray_edges = radius.get_gray_edges(
        gray_image, blur_sigma=max(1.0, median_radius / 20),
    )

    # refine：保留第二階段的分數
    refined_circles = radius.get_refined_circles(
        detected_circles, gray_edges[1], keep_score=True,
    )
    # 標記被圖片邊界切到的圓 → [x, y, r, score, partial]
    final_circles = radius.get_partial_circles(refined_circles)

    partial_count = sum(c[4] for c in final_circles)
    print(f'第三階段：{len(final_circles)} 個圓，'
          f'其中 {partial_count} 個被邊界切到')

    # 畫出最後的結果：畫在彩色原圖上
    # 完整的圓 → 紅色；被邊界切到的圓（partial）→ 橙色
    circle_colors = [
        (0, 165, 255) if c[4] else (0, 0, 255)
        for c in final_circles
    ]
    path = radius.save_image(
        radius.color_image, 'final_circles.png',
        circle_centers=[c[:2] for c in final_circles],
        circle_radii=[c[2] for c in final_circles],
        circle_color=circle_colors,
        line_thickness=3,
    )
    print('結果圖存到：', path)
    pass
