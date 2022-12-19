## Installation

Our codebase is developed based on Ubuntu 20.04 and NVIDIA GPU cards. 

### Requirements
- Python 3.9
- Pytorch 1.11
- torchvision 0.12.0
- cuda 11.3

### Setup with Conda

We suggest to create a new conda environment and install all the relevant dependencies. 

```bash
# Create a new environment
conda create --name evrgb python=3.7
conda activate evrgb

# Install Pytorch
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch

# Install Pytorch3D
pip install matplotlib
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d

# Install requirements
pip install -r requirements.txt

# Install manopth
cd MeshGraphormer
python setup.py build develop
pip install ./manopth/.
```


