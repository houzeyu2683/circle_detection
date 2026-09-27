# 找圓的流程（01 → 02 → 03）

程式：`01_estimate_radius.py` 的 `Radius` class，完整流程寫在檔案最下面的 `if __name__=='__main__':`。

三個階段都用**同一組零件**，差別只在參數和組裝順序。

```
                 read_gray_image（用 IMREAD_GRAYSCALE 讀進原圖灰階）
                                  │
          ┌───────────────────────┼────────────────────────┐
          ▼                       ▼                        ▼
   第一階段：估計 r        第二階段：找圓            第三階段：修準
   （不知道圓多大）        （已知 r）                （回到原圖）
          │                       │                        │
          └──── r ───────────────►└──── 一批圓 ───────────►└──► 最後的圓
```

---

## 共用的中間段（第一、二階段都一樣）

```
gray_image（縮放過的灰階圖）
    │
    ├─ get_gray_edges ─────────► gray_edges = (gradient, magnitude)
    │
    ▼
get_circle_candidates          Hough 猜圓，寧可多猜         [[x, y, r], ...]
    │
    ▼
get_circle_scores（快篩）      36 個點、看寬一點            [score, ...]
get_filtered_circles           分數 ≥ 快篩門檻才留下        [[x, y, r, score], ...]
    │
    ▼
get_refined_circles            把圓心挪準                   [[x, y, r], ...]
    │
    ▼
get_circle_scores（正式）      72 個點                      [score, ...]
get_filtered_circles           分數 ≥ 正式門檻才留下        [[x, y, r, score], ...]
    │
    ▼
get_unique_circles             NMS：同一顆圓只留一個，       [[x, y, r, score], ...]
                               並換回原圖座標                （原圖座標）
```

---

## 第一階段：估計 r

還不知道圓多大，所以把圖縮成三種大小，每層都跑一次上面的中間段。
不管圓多大，總有一層剛好適合它。

```
for size_bound in (600, 300, 150):
    get_scale_ratio(size_bound)        比例 = size_bound ÷ 圖的最長邊
    （比例 > 1 就跳過，不放大）
    get_gray_image(scale_ratio)
    ── 共用的中間段 ──
    結果加進 merged_circles
                │
                ▼
get_unique_circles(merged_circles)     三層合併後再 NMS 一次
                │
                ▼
（圓少於 3 個 → 停止，估不出 r）
                │
                ▼
get_estimated_radius                   面積加權的半徑分布，最高點 = r
```

## 第二階段：找出一批圓

已經知道 r，把圖縮放到「圓的半徑 = 40」，參數對每張圖都通用。

```
get_radius_ratio(r)                    比例 = 40 ÷ r（可能縮小，也可能放大）
get_gray_image(scale_ratio)            縮小用 INTER_AREA，放大用 INTER_CUBIC
── 共用的中間段（換參數）──
                │
                ▼
detected_circles                       [[x, y, r, score], ...]（原圖座標）
```

## 第三階段：在原圖上修準

第二階段是在縮小的圖上找的，回到原圖再 refine 一次，圓心更準。

```
get_gray_image()                       不縮放 = 原圖
get_gray_edges(blur = 半徑中位數 ÷ 20)
                │
                ▼
get_refined_circles(keep_score=True)   保留第二階段的分數
                │
                ▼
get_partial_circles                    標記被圖片邊界切到的圓
                │
                ▼
final_circles                          [[x, y, r, score, partial], ...]
```

---

## 每個階段的參數

| 零件 | 參數 | 第一階段 | 第二階段 | 第三階段 |
|---|---|---|---|---|
| 縮放 | 比例 | `size_bound ÷ 最長邊`（600、300、150） | `40 ÷ r` | 1（原圖） |
| `get_gray_edges` | `blur_sigma` | 1.5 | 2.0 | 半徑中位數 ÷ 20（至少 1） |
| `get_circle_candidates` | `blur_sigma` | 1.5 | 2.0 | — |
| | `center_distance` | 4 | 10 | — |
| | `vote_threshold` | 25 | 12 | — |
| | `minimum_radius` | 6 | 24 | — |
| | `maximum_radius` | 圖的短邊 ÷ 2 | 56 | — |
| `get_circle_scores` | `strong_quantile` | 0.8 | 0.65 | — |
| | 快篩 | `point_count=36, buffer_ratio=0.06` | 同左 | — |
| `get_filtered_circles` | 快篩門檻 | 0.55 | 0.45 | — |
| | 正式門檻 | 0.7 | 0.6 | — |
| `get_refined_circles` | `keep_score` | False | False | **True** |
| `get_unique_circles` | `overlap_ratio` | 0.75 | 0.6 | — |
| | `scale_ratio` | 那一層的比例；合併時 1.0 | `40 ÷ r` | — |

沒寫的參數都用預設值。

---

## 圓的格式在流程中的變化

| 位置 | 格式 |
|---|---|
| `get_circle_candidates` 之後 | `[x, y, r]`（縮放後座標） |
| `get_filtered_circles` 之後 | `[x, y, r, score]`（縮放後座標） |
| `get_refined_circles` 之後（第一、二階段） | `[x, y, r]`，分數不保留，接著會重新打分數 |
| `get_unique_circles` 之後 | `[x, y, r, score]`（**原圖座標**） |
| `get_refined_circles` 之後（第三階段） | `[x, y, r, score]`，保留第二階段的分數 |
| `get_partial_circles` 之後 | `[x, y, r, score, partial]` |

---

## 跟 notebook 比對的結果（sample_01～04）

- 第一階段的 r：四張圖都跟 `01_estimate_radius.ipynb` 一樣
- 第二階段的圓：四張圖都跟 `02_detect_circles.ipynb` 一樣
- 第三階段的圓：數量和 partial 都一樣，座標最多差 0.42 像素。
  原因是 notebook 02 存檔時把圓四捨五入到小數點後 2 位，03 再拿四捨五入過的圓去 refine；
  這裡是直接用完整精度的圓。先四捨五入再 refine 的話，就會完全一樣。
