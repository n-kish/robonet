import torch
import torch_scatter
import sys

# Force flush print statements
def print_flush(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

print_flush("1. CUDA Setup:")
print_flush(f"PyTorch version: {torch.__version__}")
print_flush(f"CUDA available: {torch.cuda.is_available()}")
print_flush(f"CUDA version: {torch.version.cuda}")
print_flush(f"Current device: {torch.cuda.current_device()}")
print_flush(f"Device name: {torch.cuda.get_device_name()}")

print_flush("\n2. Testing basic CUDA operations:")
try:
    # Test 1: Simple tensor on CPU
    print_flush("Creating CPU tensor...")
    x_cpu = torch.randn(10)
    print_flush("CPU tensor created:", x_cpu)

    # Test 2: Moving tensor to GPU
    print_flush("Moving tensor to GPU...")
    x_gpu = x_cpu.cuda()
    print_flush("Moved to GPU successfully")

    # Test 3: Direct GPU tensor creation
    print_flush("Creating GPU tensor directly...")
    x = torch.randn(10, device='cuda')
    print_flush("Direct GPU tensor created")

    # Test 4: Index tensor
    print_flush("Creating index tensor...")
    index = torch.tensor([0, 1, 0, 1, 2, 2, 3, 3, 4, 4], device='cuda')
    print_flush("Index tensor created")

except Exception as e:
    print_flush("Error:", e)
    print_flush("Stack trace:", torch.cuda.get_device_properties(0))

# Try to reset CUDA device
try:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
except Exception as e:
    print_flush("Error resetting CUDA:", e)