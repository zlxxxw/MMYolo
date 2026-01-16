import os
from ultralytics import YOLO
import pandas as pd
import torch

# ================= 🔬 修正后的实验配置 =================
# 1. 数据集配置文件路径
DATASET_YAML = 'data.yaml'

# 2. 项目名称 (为了区分，建议改个名，或者删掉旧文件夹)
PROJECT_NAME = 'YOLO_Benchmark_Optimized'

# 3. 对比模型列表
# 建议：先只跑 YOLOv8n 验证效果，没问题了再把其他模型注释解开
models_config = {
    # 'YOLOv5n': 'yolov5n.pt',
    # 'YOLOv6n': 'yolov6n.pt',
    
    'YOLOv8n': 'yolov8n.pt',   # <--- 先跑这个验证！
    
    # 'YOLOv9t': 'yolov9t.pt',
    # 'YOLOv10n': 'yolov10n.pt',
    # 'YOLOv11n': 'yolo11n.pt'
}

# 4. 优化后的超参数 (针对微调/小数据集优化)
HYPERPARAMS = {
    # --- 系统设置 ---
    'device': 0 if torch.cuda.is_available() else 'cpu',
    'workers': 4,

    # --- 训练控制 ---
    'epochs': 100,          # 先跑100轮，不需要300
    'patience': 20,         # 20轮不提升就停止 (原50太长)
    'batch': 16,
    'imgsz': 640,
    
    # --- 核心修正：优化器与学习率 ---
    'optimizer': 'AdamW',   # 🔥 修改: 使用 AdamW，比 SGD 更稳
    'lr0': 0.001,           # 🔥 修改: 降低10倍 (原0.01)，保护预训练权重
    'lrf': 0.01,            # 最终学习率 = lr0 * lrf
    'warmup_epochs': 3.0,   # 热身轮次

    # --- 核心修正：冻结骨干 ---
    'freeze': 10,           # 🔥 修改: 冻结 Backbone 前10层，只训练 Head
                            # 这能有效解决"灾难性遗忘"问题

    # --- 数据增强 (保持默认或适当减弱) ---
    'mosaic': 1.0,
    'mixup': 0.1,           # 稍微降低 mixup
    'close_mosaic': 10,
    
    'seed': 42,
    'exist_ok': True        # 允许覆盖旧实验
}
# ========================================================

def run_comparison():
    print(f"🚀 启动优化后的实验 (AdamW + Low LR + Freeze Backbone)")
    print(f"📋 待训练模型: {list(models_config.keys())}")
    
    if not os.path.exists(DATASET_YAML):
        raise FileNotFoundError(f"❌ 错误: 找不到 {DATASET_YAML}")

    final_results = []

    for display_name, model_file in models_config.items():
        print(f"\n{'='*80}")
        print(f"🤖 正在训练: {display_name}")
        print(f"{'='*80}")
        
        try:
            # 1. 加载模型
            model = YOLO(model_file)
            
            # 2. 训练
            print(f"⚙️ 参数: lr0={HYPERPARAMS['lr0']}, opt={HYPERPARAMS['optimizer']}, freeze={HYPERPARAMS['freeze']}")
            model.train(
                data=DATASET_YAML,
                project=PROJECT_NAME,
                name=display_name,
                **HYPERPARAMS
            )
            
            # 3. 验证
            print(f"📊 验证中...")
            metrics = model.val(split='val', verbose=False)
            
            # 4. 记录结果
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
            print(f"✅ {display_name} 完成! mAP50: {metrics.box.map50:.4f}")

        except Exception as e:
            print(f"❌ {display_name} 训练失败: {e}")
            import traceback
            traceback.print_exc()

    # 5. 输出最终报表
    if final_results:
        df = pd.DataFrame(final_results)
        df = df.sort_values(by='mAP50', ascending=False)
        
        csv_path = os.path.join(PROJECT_NAME, 'Optimized_Results.csv')
        df.to_csv(csv_path, index=False)
        
        print(f"\n{'='*80}")
        print(f"🏆 优化实验结束！结果已保存至: {csv_path}")
        print(df.to_string())
    else:
        print("\n⚠️ 无结果生成")

if __name__ == '__main__':
    run_comparison()