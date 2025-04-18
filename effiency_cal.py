import os
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: %s,  CUDA_VISIBLE_DEVICES: %s\n" % (device, '2'))

import time
from fvcore.nn import FlopCountAnalysis
from Modules.models.RaiSDNet import Swin_RaiSDNet


# 创建模型并移动到设备
model = Swin_RaiSDNet(in_chans=3, num_classes=2)
model.cuda()
model.eval()

# 输入样本
input_1 = torch.randn(1, 3, 587, 587).to(device)
input_2 = torch.randn(1, 3, 587, 587).to(device)

def count_parameters(model):
    """统计模型可训练参数量(单位：百万)"""
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1e6  # 转换为百万单位

# 计算参数量
print('Caculating model parameters...')
params_m = count_parameters(model)
print(f"Params: {params_m:.2f}M")


# -------------------------
# 计算 FLOPs
# -------------------------
flops = FlopCountAnalysis(model, (input_1, input_2))
print(f"FLOPs: {flops.total() / 1e9:.2f} GFLOPs")  # 输出以 GFLOPs 为单位

# -------------------------
# 计算 FPS
# -------------------------
# 进行热身，避免第一次推理的延迟
for _ in range(10):
    _ = model(input_1, input_2)

# 开始计时
torch.cuda.synchronize()  # 确保 GPU 完成计算
start_time = time.time()

# 进行多次推理并测量时间
num_runs = 100  # 设置推理次数
for _ in range(num_runs):
    _ = model(input_1, input_2)

torch.cuda.synchronize()  # 确保 GPU 完成计算
end_time = time.time()

# 计算 FPS
fps = num_runs / (end_time - start_time)
print(f"FPS: {fps:.2f} frames/second")
