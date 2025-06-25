
#!/bin/bash

# Create Conda environment from YAML file
echo "Creating Conda environment..."
conda env create -f environment.yaml

# Activate the Conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate m2diffuser

# Install scikit-learn via Conda
echo "Installing scikit-learn via Conda..."
conda install -y scikit-learn 

# Install system dependencies
echo "Installing system dependency: libsuitesparse-dev..."
sudo apt-get update
sudo apt-get install -y libsuitesparse-dev

# Install PyTorch with CUDA 12.1 (no official 12.4 wheels as of now)
echo "Installing PyTorch 2.2.2 with CUDA 12.1..."
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

# Install grasp_diffusion in editable mode
echo "Installing grasp_diffusion..."
cd third_party/grasp_diffusion
export CUDA_HOME=/usr/local/cuda
pip install -e .

# Reinstall PyTorch to ensure compatibility
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

# Install pytorch3d compatible with current PyTorch
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.5"

# Install PointNet2 ops
cd ../Pointnet2_PyTorch/pointnet2_ops_lib/
echo "Installing Pointnet2 ops..."
pip install .

# Install pointops from GitHub
echo "Installing pointops from GitHub..."
pip install "git+https://github.com/Silverster98/pointops"

# Return to the project root directory
cd ../../../

# Install Python package requirements
echo "Installing Python requirements..."
pip install -r requirement.txt

# Install PyTorch Lightning
echo "Installing PyTorch Lightning..."
pip install pytorch-lightning==2.0.0

# Install NVIDIA Kaolin (only CUDA 12.1 wheels are available)
echo "Installing NVIDIA Kaolin..."
pip install kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.2.2_cu121.html

# Install mesh2sdf
echo "Installing mesh2sdf..."
pip install mesh2sdf -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "Environment setup complete!"

