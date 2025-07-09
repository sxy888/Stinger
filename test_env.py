import tensorflow as tf
import torch
import numpy as np

def validate_tensorflow_gpu():
    print("="*50)
    print("TensorFlow-GPU 安装验证")
    print("="*50)
    
    # 1. 版本信息
    print(f"TensorFlow 版本: {tf.__version__}")
    
    # 2. GPU 可用性
    gpu_available = tf.test.is_gpu_available()
    print(f"GPU 可用: {gpu_available}")
    
    if gpu_available:
        # 3. GPU 设备信息
        gpus = tf.config.experimental.list_physical_devices('GPU')
        print(f"检测到 {len(gpus)} 个 GPU 设备")
        
        # 4. CUDA 和 cuDNN 版本（TF 1.x）
        print(f"CUDA 版本: {tf.test.is_built_with_cuda()}")
        
        # 5. 简单的 GPU 计算测试
        print("\n进行 GPU 计算测试...")
        with tf.device('/gpu:0'):
            a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            c = tf.matmul(a, b)
        
        with tf.Session() as sess:
            result = sess.run(c)
            print("GPU 矩阵运算结果:")
            print(result)
            print("✅ GPU 计算测试成功！")
    else:
        print("❌ 未检测到可用的 GPU")
    
    print("="*50)


def validate_pytorch_gpu():
    print("="*50)
    print("PyTorch GPU 安装验证")
    print("="*50)
    
    # 基本信息
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        # CUDA 详细信息
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"cuDNN 版本: {torch.backends.cudnn.version()}")
        print(f"cuDNN 启用: {torch.backends.cudnn.enabled}")
        
        # GPU 设备信息
        gpu_count = torch.cuda.device_count()
        print(f"GPU 数量: {gpu_count}")
        
        for i in range(gpu_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  显存总量: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
        
        print(f"当前 GPU 设备: {torch.cuda.current_device()}")
        
        # GPU 计算测试
        print("\n进行 GPU 计算测试...")
        device = torch.device("cuda:0")
        
        # 创建张量并移到 GPU
        x = torch.randn(1000, 1000).to(device)
        y = torch.randn(1000, 1000).to(device)
        
        # 矩阵乘法测试
        z = torch.mm(x, y)
        print(f"GPU 张量计算结果 shape: {z.shape}")
        print(f"结果在 GPU 上: {z.is_cuda}")
        print("✅ GPU 计算测试成功！")
        
    else:
        print("❌ CUDA 不可用，使用 CPU 模式")
    
    print("="*50)

# 运行验证
validate_pytorch_gpu()
    
# 运行验证
validate_tensorflow_gpu()