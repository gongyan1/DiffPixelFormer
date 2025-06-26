import matplotlib.pyplot as plt

# 从文件中读取数据
file_path = 'your_folder/DiffPixelFormer/ckpt/sunrgbd_mit_b3_pixel_att_parallel_saved/log.txt'  # 替换为你的文件路径

with open(file_path, 'r') as file:
    data = file.read()

# 初始化变量
epochs_rgb, iou_rgb = [], []
epochs_depth, iou_depth = [], []
epochs_ens, iou_ens = [], []

# 解析数据
for line in data.split('\n'):
    if "Epoch" in line:
        parts = line.split()
        epoch = int(parts[1])
        if "(rgb)" in line:
            iou_str = parts[-1] if parts[-1] != "(best)" else parts[-2]  # 获取倒数第二个或最后一个元素
            iou = float(iou_str.split('=')[1])  # 提取数值部分并转换为浮点数
            epochs_rgb.append(epoch)
            iou_rgb.append(iou)
        elif "(depth)" in line:
            iou_str = parts[-1] if parts[-1] != "(best)" else parts[-2]  # 获取倒数第二个或最后一个元素
            iou = float(iou_str.split('=')[1])  # 提取数值部分并转换为浮点数
            epochs_depth.append(epoch)
            iou_depth.append(iou)
        elif "(ens)" in line:
            iou_str = parts[-1] if parts[-1] != "(best)" else parts[-2]  # 获取倒数第二个或最后一个元素
            iou = float(iou_str.split('=')[1])  # 提取数值部分并转换为浮点数
            epochs_ens.append(epoch)
            iou_ens.append(iou)

# 绘制子图
fig, axes = plt.subplots(3, 1, figsize=(10, 18))

# 绘制RGB IoU
axes[0].plot(epochs_rgb, iou_rgb, label='RGB IoU', marker='o')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('IoU')
axes[0].set_title('RGB IoU vs. Epoch')
axes[0].legend()
axes[0].grid(True)

# 绘制Depth IoU
axes[1].plot(epochs_depth, iou_depth, label='Depth IoU', marker='x')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('IoU')
axes[1].set_title('Depth IoU vs. Epoch')
axes[1].legend()
axes[1].grid(True)

# 绘制Ens IoU
axes[2].plot(epochs_ens, iou_ens, label='Ens IoU', marker='s')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('IoU')
axes[2].set_title('Ens IoU vs. Epoch')
axes[2].legend()
axes[2].grid(True)

# 调整布局
plt.tight_layout()

# 保存图像到本地
output_path = 'iou_vs_epoch_subplots.png'  # 你可以替换为你想要保存的路径
plt.savefig(output_path)

# 显示图像
plt.show()
