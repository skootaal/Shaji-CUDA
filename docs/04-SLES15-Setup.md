# SLES15 SP7 CUDA Setup: Installation and Configuration

## System Requirements

Before installing CUDA on SUSE Linux Enterprise Server 15 SP7, ensure your system meets these requirements:

### Hardware Requirements

- **NVIDIA GPU**: CUDA Compute Capability 3.5 or higher
  - H100, A100, RTX 4090, RTX 4080, etc.
- **RAM**: Minimum 4GB (8GB recommended)
- **Disk Space**: 2-3GB for CUDA Toolkit

### Software Requirements

- **OS**: SUSE Linux Enterprise Server 15 SP7
- **Kernel**: Linux kernel 5.x or later (check with `uname -r`)
- **GCC**: GCC 9.x or later (check with `gcc --version`)
- **GLIBC**: 2.31 or later (check with `ldd --version`)

## Step 1: Verify NVIDIA GPU

### Check if GPU is detected

```bash
lspci | grep -i nvidia
```

**Expected output:**
```
3b:00.0 3D controller: NVIDIA Corporation GA100 [A100 PCIe] (rev a1)
```

### Verify GPU compute capability

```bash
# Check NVIDIA GPU Model and Specs
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

## Step 2: Install NVIDIA Display Drivers

### Method 1: Using NVIDIA CUDA Repository (Recommended)

```bash
# Add NVIDIA repository key
sudo rpm --import https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/D42D0685.pub

# Add NVIDIA CUDA repository for SLES15 SP7
sudo zypper addrepo https://developer.download.nvidia.com/compute/cuda/repos/sles15/x86_64 cuda

# Update package lists
sudo zypper refresh

# Install latest NVIDIA driver
sudo zypper install cuda-drivers
```

### Method 2: Manual Driver Installation

```bash
# Download driver for SLES15 SP7
# Visit: https://www.nvidia.com/Download/driverDetails.aspx/

# For H100 on SLES15 SP7:
wget https://us.download.nvidia.com/tesla/550.90.07/NVIDIA-Linux-x86_64-550.90.07.run

# Make executable
chmod +x NVIDIA-Linux-x86_64-550.90.07.run

# Stop X server (if running)
sudo systemctl stop display-manager

# Install driver
sudo ./NVIDIA-Linux-x86_64-550.90.07.run --dkms
```

### Verify Driver Installation

```bash
# Check driver version
nvidia-smi

# Expected output for H100:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 550.90.07              Driver Version: 550.90.07                 |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | No   Name                 Off  | 00000000:00:00.0 Off |                    0 |
# |-------------------------------+----------------------+----------------------+
# |   0  NVIDIA H100 PCIe      Off  | 00000000:00:00.0 Off |                    0 |
# | N/A   46C    P0    45W / 700W  |      0MB / 81920MB |      0B / 0B |
```

## Step 3: Install CUDA Toolkit

### Using NVIDIA Repository (Recommended)

```bash
# Update package manager
sudo zypper refresh

# Install CUDA Toolkit 12.x
sudo zypper install cuda-toolkit-12-4

# The toolkit will be installed at:
# /usr/local/cuda/
# or /usr/local/cuda-12.4/
```

### Manual Installation from RUNFILE

```bash
# Download CUDA Toolkit for SLES15 SP7
wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_560.35.03_linux.run

# Make executable
chmod +x cuda_12.4.1_560.35.03_linux.run

# Install (skip driver, use existing one)
sudo ./cuda_12.4.1_560.35.03_linux.run --toolkitpath=/usr/local/cuda-12.4 --no-questions
```

## Step 4: Configure Environment Variables

### Add to ~/.bashrc or ~/.zshrc

```bash
# CUDA paths
export CUDA_HOME=/usr/local/cuda
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Optional: For better performance
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0 (change as needed)
```

### Apply changes

```bash
source ~/.bashrc
```

## Step 5: Install Development Tools

### Install compiler and build tools

```bash
# Install GCC and build essentials
sudo zypper install gcc gcc-c++ make

# Install CUDA development files
sudo zypper install cuda-nvcc-12-4 cuda-cudart-dev-12-4

# Install cuBLAS, cuFFT, etc.
sudo zypper install cuda-cublas-12-4 cuda-cufft-12-4
```

## Step 6: Verify CUDA Installation

### Check nvcc version

```bash
nvcc --version

# Expected output:
# nvcc: NVIDIA (R) Cuda compiler driver
# Copyright (c) 2005-2023 NVIDIA Corporation
# Built on Wed_Feb__8_05:53:42_PST_2023
# Cuda compilation tools, release 12.4, V12.4.131
```

### Check GPU accessibility

```bash
# Test CUDA capability
cuda-samples

# Run deviceQuery sample
cd /usr/local/cuda/samples/1_Utilities/deviceQuery
make
./deviceQuery

# Expected output:
# CUDA Device Query (Runtime API) version (CUDART static linking)
# 
# Detected 1 CUDA Capable device(s)
# 
# Device 0: "NVIDIA H100 PCIe"
#   CUDA Driver Version / Runtime Version          12.4 / 12.4
#   CUDA Capability Major/Minor version number:    9.0
#   Total amount of global memory:                 81920 MBytes
#   GPU Clock rate:                                1980 Mhz
```

### Run bandwidthTest

```bash
cd /usr/local/cuda/samples/1_Utilities/bandwidthTest
make
./bandwidthTest

# This tests GPU-CPU data transfer speed
```

## Step 7: Troubleshooting

### Issue: "NVIDIA kernel module not loaded"

```bash
# Rebuild DKMS module
sudo dkms autoinstall

# Or manually rebuild
sudo dkms install -m nvidia -v 550.90.07

# Reboot
sudo reboot
```

### Issue: "CUDA_HOME not found"

```bash
# Find CUDA installation
find /usr -name "cuda" -type d 2>/dev/null

# Update bashrc with correct path
export CUDA_HOME=/path/to/cuda
```

### Issue: "nvcc: command not found"

```bash
# Check if nvcc is installed
which nvcc

# If not found, reinstall
sudo zypper install cuda-nvcc-12-4

# Refresh PATH
source ~/.bashrc
```

### Issue: "Permission denied" when accessing GPU

```bash
# Add user to nvidia-docker group
sudo usermod -a -G video $(whoami)
sudo usermod -a -G nvidia $(whoami)

# Relogin or run:
newgrp video
newgrp nvidia
```

## Step 8: Install cuDNN (Optional but Recommended)

For deep learning applications:

```bash
# Download cuDNN from NVIDIA (requires account)
# https://developer.nvidia.com/cudnn

# Install cuDNN
tar -xvf cudnn-*.tgz
sudo cp cudnn-*/include/cudnn.h /usr/local/cuda/include/
sudo cp cudnn-*/lib/libcudnn* /usr/local/cuda/lib64/
sudo ldconfig
```

## Step 9: Final Verification Checklist

```bash
# 1. Check driver
nvidia-smi

# 2. Check CUDA compiler
nvcc --version

# 3. Check library paths
echo $LD_LIBRARY_PATH

# 4. Verify H100 detection
lspci | grep -i nvidia

# 5. Test sample code
cd /usr/local/cuda/samples/1_Utilities/deviceQuery
make && ./deviceQuery

# 6. Check compute capability
gpu-burn 60  # Stress test for 60 seconds
```

## SLES15 SP7 Specific Notes

1. **Package Manager**: Use `zypper` instead of `apt` or `yum`
2. **Kernel**: SLES15 SP7 typically runs kernel 5.14.x or later
3. **Repository Setup**: May require adding NVIDIA repositories
4. **Firewall**: Ensure firewall doesn't block GPU access
5. **SELinux**: Disable or configure if enabled

## Quick Reference: Commands

```bash
# Check GPU
nvidia-smi

# Check driver
nvidia-smi -q | grep "Driver Version"

# Check CUDA version
nvcc --version

# Check compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

# List installed CUDA packages
zypper se cuda

# Remove CUDA
sudo zypper remove 'cuda*'

# Reinstall drivers
sudo zypper install --force cuda-drivers
```

## Next Steps

1. [Build and Execute CUDA Programs](05-Build-Execute.md)
2. [Run Sample Code Examples](../src/)
3. [Test H100 Performance](../src/h100_test.cu)
