"""Quick script to check GPU availability and configuration."""
import torch

print("="*70)
print("GPU Configuration Check")
print("="*70)

# Check CUDA availability
cuda_available = torch.cuda.is_available()
print(f"\nCUDA Available: {cuda_available}")

if cuda_available:
    print(f"CUDA Device Count: {torch.cuda.device_count()}")
    print(f"Current Device: {torch.cuda.current_device()}")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"cuDNN Version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
    
    # Check memory
    print(f"\nGPU Memory:")
    print(f"  Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"  Cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    
    print("\n[OK] GPU is ready for training!")
    print("   Training will automatically use GPU.")
else:
    print("\n[WARNING] CUDA not available. Training will use CPU.")
    print("   Make sure you have:")
    print("   1. NVIDIA GPU with CUDA support")
    print("   2. CUDA toolkit installed")
    print("   3. PyTorch with CUDA support")
    print("   Install with: pip install torch --index-url https://download.pytorch.org/whl/cu118")

print("\n" + "="*70)

