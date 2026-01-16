import os
from ultralytics import YOLO
import pandas as pd

# ================= 🔬 论文实验核心配置 =================
# 1. 数据集配置文件路径 (指向你项目根目录下的 data.yaml)
DATASET_YAML = 'data.yaml'

# 2. 实验项目名称 (结果会保存在这个文件夹下)
PROJECT_NAME = 'YOLO_8_Models_Benchmark'

# 3. 对比模型列表 (已根据你的要求更新为 8 个模型)
# 注：为了保证科研对比的严谨性，v5 使用 u 版 (Anchor-free)，v11 使用官方标准名
models_config = {
    # --- YOLOv5 系列 (经典对照) ---
    'YOLOv5n': 'yolov5n.pt',   # Nano版 (Anchor-free 架构，与 v8 对齐)
    'YOLOv5s': 'yolov5s.pt',   # Small版 (验证参数量增加带来的收益)

    # --- YOLOv6 系列 (工业推理 FPS) ---
    'YOLOv6n': 'yolov6n.pt',    # 工业界常用的高 FPS 模型 (注: 若官方源未提供会自动跳过)

    # --- YOLOv8 系列 (核心基准) ---
    'YOLOv8n': 'yolov8n.pt',    # 广泛使用的标准 Nano
    'YOLOv8s': 'yolov8s.pt',    # 标准 Small (同代模型不同量级对比)

    # --- YOLOv9 系列 (架构优化) ---
    'YOLOv9t': 'yolov9t.pt',    # PGI 架构 (t 代表 tiny，对应 nano 级)

    # --- YOLOv10 系列 (无NMS) ---
    'YOLOv10n': 'yolov10n.pt',  # 端到端检测，无需 NMS 后处理

    # --- YOLOv11 系列 (最新 SOTA) ---
    'YOLOv11n': 'yolo11n.pt'    # 2024/2025 最新版本 (官方文件名为 yolo11n)
}

# 4. 科研级超参数 (所有参数在此统一管理，防止重复报错)
HYPERPARAMS = {
    # --- 系统设置 ---
    'device': 0,            # <--- 显卡 ID 在这里统一设置 (解决报错的关键)
    'workers': 4,           # 数据加载线程 (如果卡在 Scanning 0%，请改为 0)

    # --- 基础训练参数 ---
    'epochs': 300,          # 300轮保证收敛
    'patience': 50,         # 早停机制：50轮不涨点即停止
    'batch': 16,            # 小样本推荐 16 (显存够可尝试 32)
    'imgsz': 640,           # 标准输入尺寸
    'optimizer': 'auto',    # 自动选择 (通常为 SGD)
    'seed': 42,             # 固定随机种子，保证论文可复现
    
    # --- 优化器与学习率 ---
    'lr0': 0.01,            # 初始学习率
    'lrf': 0.01,            # 最终学习率
    'cos_lr': True,         # 开启余弦退火
    'momentum': 0.937,      # 动量
    'weight_decay': 0.0005, # 权重衰减 (防止过拟合)
    'warmup_epochs': 3.0,   # 预热轮次

    # --- 增强策略 (针对小目标优化) ---
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    'degrees': 10.0,
    'translate': 0.1,
    'scale': 0.8,           # [关键] 缩放增强：让小目标变大
    'fliplr': 0.5,
    'mosaic': 1.0,          # [必开] Mosaic 增强
    'mixup': 0.15,          # [推荐] Mixup 增强
    'erasing': 0.4,         # [推荐] 随机擦除 (模拟遮挡)
    'close_mosaic': 10,     # 最后10轮关闭Mosaic，微调模型
}
# ========================================================

def run_comparison():
    print(f"🚀 开始 8 模型全方位对比实验: {list(models_config.keys())}")
    
    # 检查 yaml 文件是否存在
    if not os.path.exists(DATASET_YAML):
        raise FileNotFoundError(f"❌ 错误: 在当前目录下找不到 {DATASET_YAML} 文件！请确认路径。")

    final_results = []

    for display_name, model_file in models_config.items():
        print(f"\n{'='*80}")
        print(f"🤖 正在启动: {display_name} (权重文件: {model_file})")
        print(f"{'='*80}")
        
        try:
            # 1. 加载模型 (自动下载)
            # 注意: 如果本地没有 .pt 文件，YOLO 会自动从 GitHub Release 下载
            model = YOLO(model_file)
            
            # 2. 训练
            # 关键修复: 这里不再传入 device=0，因为 **HYPERPARAMS 里已经包含了 device
            model.train(
                data=DATASET_YAML,
                project=PROJECT_NAME,
                name=display_name,
                **HYPERPARAMS
            )
            
            # 3. 验证 (Validation)
            print(f"📊 正在验证 {display_name} 最佳权重...")
            metrics = model.val(split='val', verbose=False)
            
            # 4. 记录核心数据
            info = model.info() 
            params = info[1] if info else 0
            flops = info[2] if (info and len(info)>2) else 0

            result_entry = {
                'Model': display_name,
                'mAP50': round(metrics.box.map50, 4),
                'mAP50-95': round(metrics.box.map, 4),
                'Precision': round(metrics.box.mp, 4),
                'Recall': round(metrics.box.mr, 4),
                'Params(M)': round(params / 1e6, 2), # 参数量 (百万)
                'FLOPs(G)': round(flops / 1e9, 2)    # 计算量 (十亿)
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
        # 按 mAP50 降序排列，方便直接看谁最强
        df = df.sort_values(by='mAP50', ascending=False)
        
        csv_filename = os.path.join(PROJECT_NAME, 'Final_8_Models_Benchmark.csv')
        df.to_csv(csv_filename, index=False)
        
        print(f"\n{'='*80}")
        print(f"🏆 实验全部结束！结果已保存至: {csv_filename}")
        print(df.to_string())
        print(f"{'='*80}")
    else:
        print("\n⚠️ 没有模型完成训练，请检查数据集路径或网络连接。")

if __name__ == '__main__':
    run_comparison()