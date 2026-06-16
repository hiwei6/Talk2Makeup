import cv2
import numpy as np

def gradation_smoothing(mask, kernel_size=(12, 12), iterations=1):
    """
    實作 Gradation Smoothing 以自然融合妝容邊緣。
    [修正] 加入「強度正規化」，防止細小區域（如嘴唇）因模糊而變淡消失。
    """
    if mask.dtype != np.uint8:
        # 將 0-1 float 轉為 0-255 uint8
        mask_uint8 = (mask * 255).astype(np.uint8)
    else:
        mask_uint8 = mask
        
    # 確保 kernel_size 為奇數
    k_w, k_h = kernel_size
    if k_w % 2 == 0: k_w += 1
    if k_h % 2 == 0: k_h += 1
    valid_kernel_size = (k_w, k_h)

    # 1. Dilation (擴張)
    if iterations > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, valid_kernel_size)
        processed_mask = cv2.dilate(mask_uint8, kernel, iterations=iterations)
    else:
        processed_mask = mask_uint8
    
    # 2. Gaussian Blur (平滑邊緣)
    smoothed = cv2.GaussianBlur(processed_mask, valid_kernel_size, 0)
    
    # 3. [關鍵修正] 強制正規化 (Normalization)
    # 這一步至關重要！它會把因模糊而變暗的遮罩重新拉亮到 255
    if smoothed.max() > 0:
        smoothed_float = smoothed.astype(np.float32)
        smoothed = (smoothed_float / smoothed_float.max()) * 255.0
        
    return smoothed.astype(np.float32) / 255.0

def get_dummy_masks(image_np):
    """(Placeholder) 生成假 Mask 用於測試"""
    h, w, _ = image_np.shape
    masks = {}
    lip_mask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(lip_mask, (w//2, h//2 + 50), (40, 20), 0, 0, 360, 1, -1)
    masks['lips'] = lip_mask
    skin_mask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(skin_mask, (w//2, h//2), (100, 130), 0, 0, 360, 1, -1)
    skin_mask = np.maximum(skin_mask - lip_mask, 0)
    masks['skin'] = skin_mask
    return masks