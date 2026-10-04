import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# 1. 初始化全自動遮罩生成器
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
checkpoint = "./checkpoints/sam2.1_hiera_tiny.pt" # 請確保有下載此權重檔

device = "cuda" if torch.cuda.is_available() else "cpu"
sam2_model = build_sam2(model_cfg, checkpoint, device=device)

# 使用 AutomaticMaskGenerator，它不需要任何點擊提示
mask_generator = SAM2AutomaticMaskGenerator(sam2_model)

# 2. 讀取您的圖片
image_path = "Data/fake_sem_01.jpg" # 換成您的圖片檔名
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print("SAM 正在全自動辨識圖中所有物件，請稍候...")
# 3. 讓 SAM 自動找出圖片中「所有」東西的邊緣
masks = mask_generator.generate(image_rgb)

display_image = image.copy()

# 4. 撈出 SAM 找到的每一個物件，並用 OpenCV 計算圓心
for mask_data in masks:
    # 取出單一物件的二值化遮罩 (True/False 轉成 0/255)
    mask = mask_data["segmentation"].astype(np.uint8) * 255
    
    # 尋找輪廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 50: # 過濾掉太小的雜訊雜物
            continue
            
        # 計算幾何矩
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # 在原圖上標示出所有找到的圓心與邊緣
            cv2.drawContours(display_image, [contour], -1, (0, 255, 0), 2) # 綠色邊緣
            cv2.circle(display_image, (cx, cy), 5, (0, 0, 255), -1)      # 紅色圓心
            print(f"自動偵測到物件！圓心座標為: (X: {cx}, Y: {cy}), 面積: {area}")

# 5. 儲存或顯示最終結果
cv2.imwrite("output_result.jpg", display_image)
print("處理完成！結果已儲存至 output_result.jpg")
