import os
from ultralytics.models import YOLO
import datetime
import pandas as pd

# ================= 🔬 论文实验核心配置 =================
# 1. 数据集路径 (请务必确认正确)
DATASET_YAML = '/path/to/your/dataset/data.yaml'  # ⚠️ 修改为你的绝对路径

# 2. 实验项目名称
PROJECT_NAME = 'YOLO_Evolution_Study_500img'

# 3. 对比模型列表 (已修正为官方标准文件名)
# 注意：yolov7-tiny 不在 ultralytics 原生支持列表，建议用 v10n 代替或单独跑
models_config = {
    'YOLOv5n': 'yolov5nu.pt',  # v5 Anchor-free版，与v8/v11对比更公平
    'YOLOv8n': 'yolov8n.pt',   # 稳健的基准
    'YOLOv9t': 'yolov9t.pt',   # 引入 PGI 架构，小目标检测强
    'YOLOv10n': 'yolov10n.pt', # 清华大学无NMS版本，速度极快
    'YOLOv11n': 'yolo11n.pt'   # 2024/2025 最新 SOTA
}

# 4. 科研级超参数 (针对 500张 小样本优化)
# ================= 针对小目标检测的优化配置 =================
HYPERPARAMS = {
    # --- 基础训练参数 ---
    'epochs': 300,        # 保持300轮，给模型足够时间消化增强后的数据
    'patience': 50,       # 早停
    'batch': 16,          # 小Batch有助于BatchNorm在小数据上的表现
    'imgsz': 640,         # 如果显存够大(>12G)，强烈建议改为 1024 或 1280
    'optimizer': 'auto',
    'seed': 42,
    
    # --- 核心：几何增强 (解决小目标看不清的问题) ---
    'hsv_h': 0.015,       # 色调变化 (微调)
    'hsv_s': 0.7,         # 饱和度变化 (增强，模拟不同光照)
    'hsv_v': 0.4,         # 亮度变化 (增强，模拟阴影/强光)
    'degrees': 10.0,      # 旋转 +/- 10度 (小目标对角度敏感，不宜过大)
    'translate': 0.1,     # 平移 +/- 10%
    'scale': 0.8,         # [重点] 缩放增益。0.8意味着图像可能被放大很多。
                          # 放大 = 小目标变大 = 更容易被检测到！
    'shear': 0.0,         # 剪切 (建议关闭，容易把小目标扭曲变形)
    'perspective': 0.0005,# 透视变换 (微量，模拟摄像头角度倾斜)
    'flipud': 0.0,        # 上下翻转 (一般关闭，除非你的目标在空中倒着飞)
    'fliplr': 0.5,        # 左右翻转 (开启，增加数据多样性)

    # --- 核心：Mosaic与Mixup (解决背景过拟合) ---
    'mosaic': 1.0,        # [重点] 必须开启 (1.0)。将4张图拼成1张，极大丰富背景。
    'mixup': 0.15,        # [重点] 开启 (0.15)。两张图透明度叠加，模拟遮挡情况。
    'copy_paste': 0.3,    # [重点] 如果有分割数据，这是神技；如果是纯框，效果减半但仍可用。
    'auto_augment': 'randaugment', # 自动增强策略
    'erasing': 0.4,       # [重点] 随机擦除40%的框。强迫模型通过局部特征识别物体（防遮挡）。
    'crop_fraction': 1.0, # 不进行中心裁剪，保留全图信息
}
# ========================================================

def run_comparison():
    print(f"🚀 开始 5 模型对比实验: {list(models_config.keys())}")
    
    # 用于存储最终结果的列表
    final_results = []

    for display_name, model_file in models_config.items():
        print(f"\n{'='*60}")
        print(f"🤖 正在训练模型: {display_name} (加载权重: {model_file})")
        print(f"{'='*60}")
        
        try:
            # 1. 加载模型 (自动下载)
            model = YOLO(model_file)
            
            # 2. 训练 (Training)
            train_results = model.train(
                data=DATASET_YAML,
                project=PROJECT_NAME,
                name=display_name,  # 结果文件夹名：YOLOv5n, YOLOv11n...
                device=0,           # 指定GPU
                **HYPERPARAMS       # 传入超参数
            )
            
            # 3. 验证 (Validation) - 获取纯净的验证集指标
            print(f"📊 正在验证 {display_name} ...")
            metrics = model.val(split='val', verbose=False)
            
            # 4. 记录核心指标
            result_entry = {
                'Model': display_name,
                'mAP50': round(metrics.box.map50, 4),
                'mAP50-95': round(metrics.box.map, 4),
                'Precision': round(metrics.box.mp, 4),
                'Recall': round(metrics.box.mr, 4),
                'Fitness': round(metrics.box.fitness, 4),
                'Parameters': model.info()[1] if model.info() else 'N/A' # 记录参数量
            }
            final_results.append(result_entry)
            print(f"✅ {display_name} 完成! mAP50: {metrics.box.map50:.3f}")

        except Exception as e:
            print(f"❌ 模型 {display_name} 训练失败: {e}")

    # ================= 结果汇总与保存 =================
    if final_results:
        # 导出为 CSV 表格
        df = pd.DataFrame(final_results)
        csv_path = os.path.join(PROJECT_NAME, 'Final_Comparison_Table.csv')
        df.to_csv(csv_path, index=False)
        
        print(f"\n{'='*60}")
        print("🏆 最终对比结果 (已保存至 CSV):")
        print(df.to_string())
        print(f"{'='*60}")
        print(f"📂 所有训练图表保存在文件夹: ./{PROJECT_NAME}/")

if __name__ == '__main__':
    run_comparison()