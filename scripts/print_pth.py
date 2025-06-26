import torch

# 加载 .pth 文件
checkpoint = torch.load('your_folder/DiffPixelFormer/ckpt/nyudv2_mit_b3_pixel_att_parallel_diff_softmax_with_sclar_sum1_fast_demo/checkpoint.pth.tar')

# 打印所有字典的键
segformer_checkpoint = checkpoint["segmenter"]

# with open("your_folder/DiffPixelFormer/scripts/1.txt", 'w') as f:

for key in segformer_checkpoint.keys():
    if "scale_factor_0_to_1" in key or "scale_factor_1_to_0" in key:
        print(key, ":", segformer_checkpoint[key])
