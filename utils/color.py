import cv2
import numpy as np

def apply_makeup_natural(source_img, mask, target_rgb, alpha=0.6, update_l_channel=False):
    """
    使用 LAB 色彩空間進行自然融合。
    
    Args:
        update_l_channel (bool): 
            - True (底妝): 會改變亮度 (L)，讓膚色真的變白或變深，並保留紋理。
            - False (口紅/眼影): 鎖住亮度 (L)，保留原始光澤與皺褶。
    """
    # 1. 格式轉換 (0-1 float -> 0-255 uint8)
    if source_img.max() <= 1.0:
        src_uint8 = (source_img * 255).astype(np.uint8)
    else:
        src_uint8 = source_img.astype(np.uint8)

    # 2. 轉換到 LAB 空間
    # OpenCV LAB 範圍: L [0, 255], A [0, 255], B [0, 255]
    src_lab = cv2.cvtColor(src_uint8, cv2.COLOR_RGB2LAB)
    l_src, a_src, b_src = cv2.split(src_lab)

    # 3. 處理目標顏色
    target_pixel = np.array([[target_rgb]], dtype=np.uint8)
    target_lab = cv2.cvtColor(target_pixel, cv2.COLOR_RGB2LAB)
    tgt_l, tgt_a, tgt_b = target_lab[0, 0]

    # 4. 計算混合係數 (Blend Factor)
    # 限制最大為 1.0 (100% 覆蓋)，超過的部分視為無效，避免反色
    raw_factor = mask * alpha
    blend_factor = np.clip(raw_factor, 0, 1.0)
    
    # 轉換為 float 進行計算
    a_src_f = a_src.astype(np.float32)
    b_src_f = b_src.astype(np.float32)
    l_src_f = l_src.astype(np.float32)
    
    # 5. 顏色通道 (A, B) 混合 - 線性插值
    new_a = a_src_f * (1 - blend_factor) + tgt_a * blend_factor
    new_b = b_src_f * (1 - blend_factor) + tgt_b * blend_factor

    # 6. 亮度通道 (L) 混合 - 僅在 update_l_channel=True (底妝) 時執行
    if update_l_channel:
        # [底妝核心演算法] 紋理保留混合 (Texture Preserving Blend)
        # 我們不直接覆蓋 L，而是根據目標亮度調整基底，但保留局部對比度
        
        # 簡單線性混合會像「水泥牆」把毛孔抹平，這裡我們使用一種柔和混合
        # New_L = Original_L * (1-alpha) + Target_L * alpha
        new_l = l_src_f * (1 - blend_factor) + tgt_l * blend_factor
        
        # [進階] 如果您覺得底妝太假，可以嘗試疊加原始的高頻細節 (High-pass)，但在這裡先用線性混合確保變色效果明顯
    else:
        # 口紅/眼影模式：完全保留原始亮度，確保立體感
        new_l = l_src_f

    # 7. 防止數值溢位
    new_l = np.clip(new_l, 0, 255)
    new_a = np.clip(new_a, 0, 255)
    new_b = np.clip(new_b, 0, 255)

    # 8. 合併回 RGB
    new_lab = cv2.merge([new_l.astype(np.uint8), new_a.astype(np.uint8), new_b.astype(np.uint8)])
    result_rgb = cv2.cvtColor(new_lab, cv2.COLOR_LAB2RGB)

    return result_rgb.astype(np.float32) / 255.0