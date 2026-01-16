import os
from ultralytics import YOLO
import torch

# ================= 🔬 最终修正版：高分+冻结策略 =================
DATASET_YAML = 'data.yaml'
PROJECT_NAME = 'YOLO_HighRes_Fix'

# 仅对比最有希望的两个模型
models_config = {
    'YOLOv8n': 'yolov8n.pt',
    'YOLOv8s': 'yolov8s.pt', # s版参数多一点，也许对小目标更敏感
}

HYPERPARAMS = {
    # --- 核心改变：分辨率与显存平衡 ---
    'imgsz': 1280,          # ⬆️ 关键：提升分辨率，让小目标从3像素变成10像素
    'batch': 8,             # ⬇️ 降低Batch以适应1280分辨率 (显存若够大可试8)
    
    # --- 训练策略 ---
    'epochs': 150,          # 150轮足够看清趋势
    'patience': 30,         # 早停
    'optimizer': 'SGD',     # ✅ 回归最稳的 SGD
    'lr0': 0.01,
    'lrf': 0.01,
    'momentum': 0.937,
    
    # --- 迁移学习策略 ---
    'freeze': 10,           # ❄️ 冻结骨干：防止500张图带偏整个模型
                            # 强迫头部(Head)去适应骨干提取的特征
    
    # --- 温和的数据增强 ---
    'mosaic': 0.5,          # ⬇️ 降低马赛克概率 (之前是1.0)，减少小目标被切碎的风险
    'mixup': 0.1,           # 轻微混合
    'scale': 0.5,           # 缩放范围
    'degrees': 0.0,         # 关闭旋转 (人倒过来这种场景很少)
    'close_mosaic': 20,     # 最后20轮关闭增强，进行精细微调
    
    # --- 系统 ---
    'device': 0,
    'workers': 0,           # 保持0防死锁
    'exist_ok': True
}

def run_final_fix():
    print(f"🚀 启动最终修正实验: 1280分辨率 + 冻结骨干 + 温和增强")
    
    if not os.path.exists(DATASET_YAML):
        raise FileNotFoundError("找不到 data.yaml")

    for display_name, model_file in models_config.items():
        print(f"\n{'='*60}")
        print(f"🔥 正在训练: {display_name} (ImgSz: 1280)")
        print(f"{'='*60}")
        
        try:
            model = YOLO(model_file)
            
            # 训练
            results = model.train(
                data=DATASET_YAML,
                project=PROJECT_NAME,
                name=display_name,
                **HYPERPARAMS
            )
            
            # 验证 (使用同样的大分辨率)
            print(f"📊 验证 {display_name} ...")
            metrics = model.val(split='val', imgsz=1280)
            
            print(f"✅ {display_name} 结果:")
            print(f"   mAP50:    {metrics.box.map50:.4f}")
            print(f"   Precision:{metrics.box.mp:.4f}")
            print(f"   Recall:   {metrics.box.mr:.4f}")

        except Exception as e:
            print(f"❌ 训练失败: {e}")
            print("💡 提示：如果报 CUDA OOM (显存不足)，请去脚本里把 'batch': 4 改为 'batch': 2")

if __name__ == '__main__':
    run_final_fix()