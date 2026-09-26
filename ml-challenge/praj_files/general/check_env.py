import sys
import os
import platform

def print_separator(title):
    print(f"\n{'='*20} {title} {'='*20}")

def check_system():
    print_separator("System Information")
    print(f"OS Platform: {platform.platform()}")
    print(f"Python Version: {sys.version.split(' ')[0]}")
    print(f"Working Directory: {os.getcwd()}")
    
    # Check RAM
    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.readlines()
            total_ram = [line for line in meminfo if 'MemTotal' in line][0].split()[1]
            print(f"Total RAM: {int(total_ram) / (1024**2):.2f} GB")
    except:
        print("Could not read /proc/meminfo for RAM")

def check_pytorch_and_cuda():
    print_separator("PyTorch & CUDA Information")
    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Version (built with): {torch.version.cuda}")
            num_gpus = torch.cuda.device_count()
            print(f"Number of GPUs: {num_gpus}")
            for i in range(num_gpus):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)} (VRAM: {props.total_memory / (1024**3):.2f} GB)")
    except ImportError:
        print("PyTorch is not installed!")

def check_packages():
    print_separator("Key ML Packages")
    packages_to_check = [
        "transformers", "accelerate", "bitsandbytes", 
        "peft", "datasets", "trl", "cudf", "pandas", "sklearn"
    ]
    
    import importlib
    for pkg in packages_to_check:
        try:
            module = importlib.import_module(pkg)
            version = getattr(module, "__version__", "Unknown Version")
            print(f"[OK] {pkg}: {version}")
        except ImportError:
            print(f"[MISSING] {pkg}")

def check_challenge_data():
    print_separator("Challenge Dataset")
    dataset_dir = "business_entity_resolution/challenge-dataset"
    if os.path.exists(dataset_dir):
        print(f"Dataset directory found at: {dataset_dir}")
        print("Contents:")
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                if file.endswith('.tsv'):
                    filepath = os.path.join(root, file)
                    size_mb = os.path.getsize(filepath) / (1024 * 1024)
                    print(f"  - {os.path.relpath(filepath, dataset_dir)} ({size_mb:.2f} MB)")
    else:
        print(f"Dataset directory NOT found at: {dataset_dir}")
        print("Please ensure it is mounted correctly via Docker.")

if __name__ == "__main__":
    print("Gathering environment information...")
    check_system()
    check_pytorch_and_cuda()
    check_packages()
    check_challenge_data()
    print_separator("Done")
