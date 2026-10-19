import torch
import numpy as np
import cv2
import torchvision.transforms as transforms
from PIL import Image
from models.bisenet import BiSeNet

# 定義顏色表 (19類)
PART_COLORS = [
    [0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255],
    [204, 0, 204], [0, 255, 255], [255, 204, 204], [102, 51, 0], [255, 0, 0],
    [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204], [255, 51, 153],
    [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]
]

def load_bisenet(weights_path, device="cuda"):
    """載入 BiSeNet 模型"""
    net = BiSeNet(n_classes=19)
    net.to(device)
    
    if torch.cuda.is_available():
        state_dict = torch.load(weights_path, map_location='cuda')
    else:
        state_dict = torch.load(weights_path, map_location='cpu')
    
    # 處理 module. 前綴
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    # 自動偵測 yakhyo 命名規則並轉換
    final_state_dict = {}
    for k, v in new_state_dict.items():
        if k.startswith('fpn.backbone.'):
            k = k.replace('fpn.backbone.', 'cp.resnet.')
        elif k.startswith('fpn.'):
            k = k.replace('fpn.', 'cp.')
        final_state_dict[k] = v

    try:
        net.load_state_dict(final_state_dict, strict=False)
        print("✅ BiSeNet weights loaded successfully.")
    except RuntimeError as e:
        print(f"⚠️ Weight mismatch warning: {e}")
        
    net.eval()
    return net

def visualize_parsing_maps(img, parsing_anno):
    """將分割結果視覺化"""
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    for pi in range(1, 19):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = PART_COLORS[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    
    vis_im = cv2.addWeighted(img, 0.4, vis_parsing_anno_color, 0.6, 0)
    return vis_im

def get_face_masks(img_pil, net, device="cuda"):
    """取得高精度的分割遮罩 (自動適應模型精度)"""
    w_orig, h_orig = img_pil.size
    
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    with torch.no_grad():
        image = img_pil.resize((512, 512), Image.BILINEAR)
        img = to_tensor(image)
        
        # [關鍵修正] 自動抓取模型權重的 dtype (float32 或 float16)
        # 解決 "Expected tensor for argument #1 ... but type FloatTensor does not equal HalfTensor"
        model_dtype = next(net.parameters()).dtype
        img = torch.unsqueeze(img, 0).to(device, dtype=model_dtype)
        
        out = net(img)[0]
        
        # 放大回原始尺寸以保持邊緣平滑
        out = torch.nn.functional.interpolate(
            out.unsqueeze(0), 
            size=(h_orig, w_orig), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
        
        parsing = out.cpu().numpy().argmax(0)

    # 縮放回 512x512 供 Diffusion 使用
    if (w_orig, h_orig) != (512, 512):
         parsing_512 = cv2.resize(parsing.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST)
    else:
         parsing_512 = parsing

    masks = {
        'full_parse': parsing_512,
        'eyes': np.where((parsing_512 == 4) | (parsing_512 == 5), 1.0, 0.0),
        'lips': np.where((parsing_512 == 12) | (parsing_512 == 13), 1.0, 0.0)
    }
    return masks

def get_skin_mask(parsing, h, w, include_neck=True):
    """產生底妝遮罩"""
    # 1. 皮膚 + 鼻子
    mask = np.where((parsing == 1) | (parsing == 10), 1.0, 0.0)
    
    # 2. 加入脖子
    if include_neck:
        neck = np.where((parsing == 14) | (parsing == 15), 1.0, 0.0)
        mask = np.maximum(mask, neck)
        
    # 3. 排除五官
    features = np.where(
        (parsing == 2) | (parsing == 3) | # 眉毛
        (parsing == 4) | (parsing == 5) | # 眼睛
        (parsing == 11) | (parsing == 12) | (parsing == 13), # 嘴巴
        1.0, 0.0
    )
    mask = cv2.subtract(mask, features)
    
    # 4. 邊緣羽化
    mask = cv2.GaussianBlur(mask, (25, 25), 0)
    return np.clip(mask, 0, 1)

def get_advanced_masks(parsing, h, w, blush_size=1.0):
    """
    生成精細遮罩 (包含眉毛、嘴唇、腮紅、底妝)

    Args:
        blush_size (float): 腮紅範圍縮放係數 (由 LLM 判斷或手動調整)。
            1.0 = 預設範圍，<1.0 縮小暈染範圍（如「小範圍蘋果肌腮紅」），
            >1.0 放大暈染範圍（如「大範圍打亮腮紅」）。
    """
    # 1. 眉毛
    brows = np.where((parsing == 2) | (parsing == 3), 1.0, 0.0)

    # 2. 嘴唇 (僅去除牙齒，暫時不內縮)
    # 這樣可以確保遮罩面積最大化，一定看得到顏色
    lips_raw = np.where((parsing == 12) | (parsing == 13), 1.0, 0.0).astype(np.uint8)
    mouth_inner = np.where(parsing == 11, 1.0, 0.0).astype(np.uint8) # 牙齒

    # 只做減法，不做 erode
    lips_final = cv2.subtract(lips_raw, mouth_inner).astype(np.float32)

    # 3. 腮紅 (幾何計算，範圍隨 blush_size 彈性縮放)
    blush_size = float(np.clip(blush_size, 0.4, 2.5))
    blush_mask = np.zeros((h, w), dtype=np.float32)
    l_eye = np.argwhere(parsing == 4)
    r_eye = np.argwhere(parsing == 5)

    if len(l_eye) > 0 and len(r_eye) > 0:
        ly, lx = l_eye.mean(axis=0)
        ry, rx = r_eye.mean(axis=0)
        offset_y = h * 0.12
        offset_x = (rx - lx) * 0.15

        seed_radius = max(1, int(round(5 * blush_size)))
        cv2.circle(blush_mask, (int(lx - offset_x), int(ly + offset_y)), seed_radius, 1, -1)
        cv2.circle(blush_mask, (int(rx + offset_x), int(ry + offset_y)), seed_radius, 1, -1)

        # 暈染半徑跟著 blush_size 縮放：範圍越大，色塊邊界越柔和、覆蓋越廣
        blur_size = max(3, int(w / 3 * blush_size)) | 1
        blush_mask = cv2.GaussianBlur(blush_mask, (blur_size, blur_size), 0)
        if blush_mask.max() > 0: blush_mask /= blush_mask.max()

        skin = np.where(parsing == 1, 1.0, 0.0)
        nose = np.where(parsing == 10, 1.0, 0.0)
        nose_dilated = cv2.dilate(nose.astype(np.uint8), np.ones((15,15), np.uint8), iterations=1)
        blush_mask = blush_mask * skin * (1 - nose_dilated)

    # 4. 底妝
    skin_mask = get_skin_mask(parsing, h, w, include_neck=True)

    return {
        "lips": lips_final,
        "brows": brows,
        "blush": blush_mask,
        "skin": skin_mask,
        "parsing": parsing
    }

def generate_realistic_eyeshadow(parsing, h, w, size_scale=1.0):
    """生成高擬真的漸層眼影遮罩

    Args:
        size_scale (float): 暈染範圍縮放係數 (由 LLM 判斷或手動調整)。
            1.0 = 預設範圍，<1.0 收斂成「眼頭/眼尾小範圍提亮」，
            >1.0 放大成「大範圍煙燻眼」。
    """
    size_scale = float(np.clip(size_scale, 0.4, 2.5))
    l_eye = np.where(parsing == 4, 1.0, 0.0)
    r_eye = np.where(parsing == 5, 1.0, 0.0)
    brows = np.where((parsing == 2) | (parsing == 3), 1.0, 0.0)

    eyeshadow_mask = np.zeros((h, w), dtype=np.float32)

    for eye_class, direction in [(4, 'left'), (5, 'right')]:
        eye_mask = np.where(parsing == eye_class, 1.0, 0.0).astype(np.uint8)
        if eye_mask.sum() == 0: continue

        shift_y = int(h * 0.02 * size_scale)
        shift_x = int(w * 0.015 * size_scale)
        M = np.float32([[1, 0, 0], [0, 1, 0]])

        shift_x_dir = -shift_x if direction == 'left' else shift_x
        M[1, 2] = -shift_y
        M[0, 2] = shift_x_dir

        kernel_size = (max(1, int(w*0.05*size_scale)), max(1, int(h*0.04*size_scale)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        dilated_eye = cv2.dilate(eye_mask, kernel, iterations=3)
        shadow_base = cv2.warpAffine(dilated_eye, M, (w, h))
        
        shadow_base = cv2.subtract(shadow_base, eye_mask)
        shadow_base = cv2.subtract(shadow_base, brows.astype(np.uint8))
        skin = np.where(parsing == 1, 1.0, 0.0).astype(np.uint8)
        shadow_base = cv2.bitwise_and(shadow_base, skin)
        
        dist_map = cv2.distanceTransform(shadow_base, cv2.DIST_L2, 5)
        if dist_map.max() > 0: dist_map = dist_map / dist_map.max()
        dist_map = np.power(dist_map, 0.5) 
        eyeshadow_mask = np.maximum(eyeshadow_mask, dist_map)

    eyeshadow_mask = cv2.GaussianBlur(eyeshadow_mask, (15, 15), 0)
    return np.clip(eyeshadow_mask, 0, 1)