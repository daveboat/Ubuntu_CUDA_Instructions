Installing CUDA 10.0 and CUDnn 7.6 on Ubuntu 18.04, for pytorch (~1.3) and tensorflow (~2.0, 1.14-1.15) to both be able to use the GPU
Also git, anaconda, pytorch, tensorflow, etc

## 1. Install nvidia drivers if they aren't installed already

- Add nvidia PPA
```
sudo add-apt-repository ppa:graphics-drivers
sudo apt-get update
```
- Install latest nvidia driver for the graphics card
```
sudo apt-get install nvidia-driver-430
```
- Reboot
```
sudo reboot
```
- Check nvidia driver is installed
```
nvidia-smi
```
## 2. Install CUDA

- Install dependencies
```
sudo apt-get install freeglut3 freeglut3-dev libxi-dev libxmu-dev
```
- Choose the correct downloads from https://developer.nvidia.com/cuda-downloads
```
wget https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_410.48_linux
wget http://developer.download.nvidia.com/compute/cuda/10.0/Prod/patches/1/cuda_10.0.130.1_linux.run
```
- Install CUDA 10.0 and patch (Don't install driver, since we already installed the driver)
```
sudo sh cuda_10.0.130_410.48_linux
sudo sh cuda_10.0.130.1_linux.run
```
- Add CUDA directories to PATH variable
```
echo 'export PATH=/usr/local/cuda-10.0/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
sudo ldconfig
```
- Check CUDA is installed
```
nvcc --version
```
## 3. Install CUDnn

- Download the correct version of CUDnn (I got CUDnn version 7.6 for Cuda 10.0) from https://developer.nvidia.com/cudnn (requires an nVidia developer account)

- Install
```
sudo dpkg -i libcudnn7_7.6.5.32-1+cuda10.0_amd64.deb
```
## 4. Install Anaconda

- Get install script from https://www.anaconda.com/distribution/#linux
```
wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
```
- Run install script (do not sudo!)
```
sh https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
```
## 5. (Optional) Install Git
```
sudo apt install git
```
## 6. (Optional) Make a dummy conda environment, install pytorch and tensorflow, check that they can access the GPU
```
conda create -n dummy python=3.7
conda activate dummy
pip install tensorflow-gpu
pip install torch torchvision
```
```
python
>>> import torch
>>> torch.cuda.is_available()
>>> import tensorflow as tf
>>> tf.test.is_gpu_available()
```