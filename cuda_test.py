import torch

# 检查CUDA是否可用
cuda_available = torch.cuda.is_available()
if cuda_available:
    print("CUDA可用")
    # 获取CUDA版本
    cuda_version = torch.version.cuda
    print(f"CUDA版本: {cuda_version}")
else:
    print("CUDA不可用")

# 检查cuDNN是否可用
cudnn_available = torch.backends.cudnn.enabled
if cudnn_available:
    print("cuDNN可用")
    # 获取cuDNN版本
    cudnn_version = torch.backends.cudnn.version()
    print(f"cuDNN版本: {cudnn_version}")
else:
    print("cuDNN不可用")
