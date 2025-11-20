import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2, FireCLIPVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, 's2', False)
    
    # 🔥 优先检查是否为自定义 CLIP 模型
    # 支持 HuggingFace 模型名称或本地路径
    custom_clip_keywords = ['fesvhtr', 'clip-iferniu', 'custom-clip']
    is_custom_clip = any(keyword in vision_tower for keyword in custom_clip_keywords)
    
    if is_custom_clip:
        print(f'🎯 Using CustomCLIPVisionTower for: {vision_tower}')
        return FireCLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    
    # 原有的 CLIP 逻辑
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
