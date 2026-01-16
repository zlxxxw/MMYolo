import yaml
import os
import cv2
import matplotlib.pyplot as plt
import random

def verify_dataset_labels(yaml_path, num_samples=3):
    """
    随机读取训练集图片并绘制标签框，用于验证数据格式是否正确
    """
    # 1. 读取 yaml 配置
    if not os.path.exists(yaml_path):
        print(f"❌ 错误：找不到配置文件 {yaml_path}")
        return

    with open(yaml_path, 'r', encoding='utf-8') as f:
        data_cfg = yaml.safe_load(f)
    
    # 获取路径 (兼容绝对路径和相对路径)
    root_dir = data_cfg.get('path', '')
    train_dir = data_cfg.get('train', '')
    
    # 拼接完整路径
    if os.path.isabs(train_dir):
        img_dir = train_dir
    else:
        img_dir = os.path.join(root_dir, train_dir)

    print(f"📂 正在检查数据集目录: {img_dir}")
    
    # 2. 获取所有图片
    supported_ext = ['.jpg', '.jpeg', '.png', '.bmp']
    images = [f for f in os.listdir(img_dir) if os.path.splitext(f)[-1].lower() in supported_ext]
    
    if not images:
        print("❌ 未发现图片，请检查 data.yaml 中的路径配置！")
        return

    # 3. 随机抽样
    samples = random.sample(images, min(len(images), num_samples))
    
    for img_name in samples:
        img_path = os.path.join(img_dir, img_name)
        
        # 推断标签路径 (假设 labels 文件夹与 images 同级)
        # 常见结构: .../images/train/1.jpg -> .../labels/train/1.txt
        label_path = img_path.replace('images', 'labels').rsplit('.', 1)[0] + '.txt'
        
        # 读取图片
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        
        # 读取标签
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = list(map(float, line.strip().split()))
                cls_id = int(parts[0])
                # YOLO 格式: x_center, y_center, width, height (归一化 0-1)
                cx, cy, bw, bh = parts[1], parts[2], parts[3], parts[4]
                
                # 反归一化为像素坐标
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                
                # 画框
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"Class {cls_id}", (x1, y1-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            print(f"⚠️ 警告: 图片 {img_name} 没有对应的标签文件！")
            cv2.putText(img, "No Label", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 显示图片 (Matplotlib)
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"Check: {img_name}")
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    # 请确保这里指向您的 data.yaml
    verify_dataset_labels('data.yaml')