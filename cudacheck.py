import torch
import subprocess

def check_cuda():
    print("Torch version:", torch.__version__)
    cuda_available = torch.cuda.is_available()
    print("CUDA available:", cuda_available)
    if cuda_available:
        print("CUDA device count:", torch.cuda.device_count())
        print("CUDA device name:", torch.cuda.get_device_name(0))
    else:
        print("No CUDA device detected.")
    print("\n--- nvidia-smi output ---")
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
        print(result.stdout)
    except Exception as e:
        print("nvidia-smi not found or failed to run:", e)

if __name__ == "__main__":
    check_cuda()