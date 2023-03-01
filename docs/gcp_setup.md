```
sudo apt update
sudo apt install build-essential openmpi-bin wget build-essential git

wget https://developer.download.nvidia.com/compute/cuda/11.6.0/local_installers/cuda_11.6.0_510.39.01_linux.run
sudo sh cuda_11.6.0_510.39.01_linux.run

wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh
bash Anaconda3-2022.10-Linux-x86_64.sh

conda create -n flexgen python=3.9
conda activate flexgen

# (Optional) Install Lianmin's vim plugins
# git clone git@github.com:merrymercy/m-setup.git
# bash m-setup/scripts/all.sh

pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

git clone git@github.com:FMInference/FlexGen.git
cd FlexGen
pip3 install -e .

cd FlexGen/benchmark/third_party/DeepSpeed
sudo apt install libaio-dev
pip3 install -e .

cd FlexGen/benchmark/third_party/transformers
pip3 install -e .
pip3 install accelerate==0.15.0
```
