import os
from ultralytics import YOLO
import pandas as pd

# ================= 🔬 论文实验核心配置 =================
# 1. 数据集配置文件路径 (指向你项目根目录下的 data.yaml)
DATASET_YAML = 'data.yaml'

# 2. 实验项目名称 (结果会保存在这个文件夹下)
PROJECT_NAME = 'YOLO_8_Models_Benchmark_Final'

# 3. 对比模型列表 (严格按照您的要求配置)
models_config = {
    # --- YOLOv5 系列 (强制使用经典版/非u版) ---
    'YOLOv5n': 'yolov5n.pt',    # 经典 YOLOv5 Nano
    'YOLOv5s': 'yolov5s.pt',    # 经典 YOLOv5 Small

    # --- YOLOv6 系列 ---
    'YOLOv6n': 'yolov6n.pt',    # 工业界高 FPS 模型

    # --- YOLOv8 系列 ---
    'YOLOv8n': 'yolov8n.pt',    # v8 Nano
    'YOLOv8s': 'yolov8s.pt',    # v8 Small

    # --- YOLOv9 系列 ---
    'YOLOv9t': 'yolov9t.pt',    # v9 Tiny

    # --- YOLOv10 系列 ---
    'YOLOv10n': 'yolov10n.pt',  # v10 Nano (无NMS)

    # --- YOLOv11 系列 ---
    'YOLOv11n': 'yolo11n.pt'    # v11 Nano (注意：官方文件名为 yolo11n.pt)
}

# 4. 科研级超参数 (所有参数在此统一管理)
HYPERPARAMS = {
    # --- 系统设置 ---
    'device': 0,            # 显卡 ID (在此处统一设置，防止报错)
    'workers': 4,           # 数据加载线程

    # --- 基础训练参数 ---
    'epochs': 300,          # 300轮
    'patience': 50,         # 早停
    'batch': 16,            # 小样本推荐 16
    'imgsz': 640,           # 标准尺寸
    'optimizer': 'auto',    # 自动选择
    'seed': 42,             # 固定种子
    
    # --- 优化器与学习率 ---
    'lr0': 0.01,            
    'lrf': 0.01,            
    'cos_lr': True,         
    'momentum': 0.937,      
    'weight_decay': 0.0005, 
    'warmup_epochs': 3.0,   

    # --- 增强策略 (针对小目标优化) ---
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    'degrees': 10.0,
    'translate': 0.1,
    'scale': 0.8,           # 缩放增强
    'fliplr': 0.5,
    'mosaic': 1.0,          # 开启 Mosaic
    'mixup': 0.15,          # 开启 Mixup
    'erasing': 0.4,         # 开启随机擦除
    'close_mosaic': 10,     # 最后10轮关闭Mosaic
}
# ========================================================

def run_comparison():
    print(f"🚀 开始 8 模型全方位对比实验 (强制使用 v5n/v5s): {list(models_config.keys())}")
    
    # 检查 yaml 文件
    if not os.path.exists(DATASET_YAML):
        raise FileNotFoundError(f"❌ 错误: 在当前目录下找不到 {DATASET_YAML} 文件！")

    final_results = []

    for display_name, model_file in models_config.items():
        print(f"\n{'='*80}")
        print(f"🤖 正在启动: {display_name} (权重文件: {model_file})")
        print(f"{'='*80}")
        
        try:
            # 1. 加载模型
            # 若本地无文件，YOLO 会尝试自动下载
            model = YOLO(model_file)
            
            # 2. 训练
            # device 参数已包含在 HYPERPARAMS 中，此处不再重复传入
            model.train(
                data=DATASET_YAML,
                project=PROJECT_NAME,
                name=display_name,
                **HYPERPARAMS
            )
            
            # 3. 验证 (Validation)
            print(f"📊 正在验证 {display_name} 最佳权重...")
            metrics = model.val(split='val', verbose=False)
            
            # 4. 记录数据
            info = model.info() 
            params = info[1] if info else 0
            flops = info[2] if (info and len(info)>2) else 0

            result_entry = {
                'Model': display_name,
                'mAP50': round(metrics.box.map50, 4),
                'mAP50-95': round(metrics.box.map, 4),
                'Precision': round(metrics.box.mp, 4),
                'Recall': round(metrics.box.mr, 4),
                'Params(M)': round(params / 1e6, 2),
                'FLOPs(G)': round(flops / 1e9, 2)
            }
            final_results.append(result_entry)
            print(f"✅ {display_name} 结束! mAP50: {metrics.box.map50:.3f}")

        except Exception as e:
            print(f"❌ 模型 {display_name} 训练中断或不支持: {e}")
            print("⚠️ 系统将自动跳过此模型，继续执行下一个任务...")
            continue

    # 5. 保存结果
    if final_results:
        df = pd.DataFrame(final_results)
        df = df.sort_values(by='mAP50', ascending=False)
        
        csv_filename = os.path.join(PROJECT_NAME, 'Final_Benchmark_Results.csv')
        df.to_csv(csv_filename, index=False)
        
        print(f"\n{'='*80}")
        print(f"🏆 实验全部结束！结果已保存至: {csv_filename}")
        print(df.to_string())
        print(f"{'='*80}")
    else:
        print("\n⚠️ 没有任何模型完成训练，请检查网络或数据集路径。")

if __name__ == '__main__':
    run_comparison()