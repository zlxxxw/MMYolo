import os
from ultralytics import YOLO
import pandas as pd
import torch

# ================= 🚑 救援模式：朴素基线实验 =================
# 目的：排除一切干扰，验证数据本身是否能让模型收敛
DATASET_YAML = 'data.yaml'
PROJECT_NAME = 'YOLO_Rescue_Mission'

# 只跑一个最稳的模型
models_config = {
    'YOLOv8n': 'yolov8n.pt'
}

# 极简超参数 (去除所有可能导致崩溃的变量)
HYPERPARAMS = {
    # --- 系统 ---
    'device': 0 if torch.cuda.is_available() else 'cpu',
    'workers': 0,           # Windows下设为0更稳，防止死锁

    # --- 训练 ---
    'epochs': 50,           # 跑50轮足够看趋势
    'patience': 0,          # 关闭早停，强行看完曲线
    'batch': 16,
    'imgsz': 640,
    
    # --- 优化器 (回归经典) ---
    'optimizer': 'SGD',     # 换回 SGD
    'lr0': 0.01,            # 标准学习率
    'lrf': 0.01,            # 标准衰减
    'momentum': 0.937,
    
    # --- 关键：关闭所有强力增强 ---
    'mosaic': 0.0,          # ❌ 关闭马赛克 (关键!)
    'mixup': 0.0,           # ❌ 关闭混合
    'hsv_h': 0.0,           # 关闭色调变换
    'hsv_s': 0.0,
    'hsv_v': 0.0,
    'degrees': 0.0,         # 关闭旋转
    'translate': 0.0,       # 关闭平移
    'scale': 0.5,           # 保持默认缩放
    'fliplr': 0.0,          # 关闭翻转
    
    'freeze': 0,            # ❌ 彻底解冻 (让模型从头适应机场数据)
    'exist_ok': True
}

def run_rescue():
    print(f"🚀 启动救援实验: SGD + 无增强 + 全参数微调")
    
    if not os.path.exists(DATASET_YAML):
        raise FileNotFoundError("找不到 data.yaml")

    for display_name, model_file in models_config.items():
        print(f"\n{'='*40}")
        print(f"训练: {display_name}")
        print(f"{'='*40}")
        
        try:
            model = YOLO(model_file)
            model.train(
                data=DATASET_YAML,
                project=PROJECT_NAME,
                name=display_name,
                **HYPERPARAMS
            )
            print("✅ 训练完成")
            
            # 验证
            metrics = model.val(split='val')
            print(f"结果: mAP50={metrics.box.map50:.4f}")
            
        except Exception as e:
            print(f"❌ 失败: {e}")

if __name__ == '__main__':
    run_rescue()