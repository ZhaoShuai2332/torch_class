# filename: check_device.py

import torch

def main():
    # 检查 CUDA 是否可用
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")

    # 获取当前使用的设备
    device = torch.device("cuda" if cuda_available else "cpu")
    print(f"Current device: {device}")

    # 如果有 GPU，打印相关信息
    if cuda_available:
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"Current CUDA device index: {torch.cuda.current_device()}")

    # 创建一个张量并将其移动到当前设备
    tensor = torch.randn(3, 3).to(device)
    print(f"Tensor device: {tensor.device}")
    print(f"Tensor contents:\n{tensor}")

    # 打印显存使用情况
    if cuda_available:
        print("\nCUDA Memory Summary:")
        print(torch.cuda.memory_summary())

if __name__ == "__main__":
    main()
