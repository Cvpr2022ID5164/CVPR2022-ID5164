# GAML: Geometry-Aware Meta-Learner for Cross-Category 6D Pose Estimation
## Installation
- Install CUDA 10.2
- Set up python3 environment from requirement.txt:
  ``` shell
  conda create -n gaml python=3.6 pip
  conda activate gaml
  conda config --append channels conda-forge     
  conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch    
  pip3 install -r requirement.txt
  bash install_pytorch_geometric.sh    
  ```
 -  Install [apex](https://github.com/NVIDIA/apex):
  ```shell
  git clone https://github.com/NVIDIA/apex      
  cd apex      
  export TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.5"  # set the target architecture manually, suggested in issue https://github.com/NVIDIA/apex/issues/605#issuecomment-554453001        
  pip3 install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./      
  cd ..     
  ```
- Install [normalSpeed](https://github.com/hfutcgncas/normalSpeed), a fast and light-weight normal map estimator:
  ```shell
  git clone https://github.com/hfutcgncas/normalSpeed.git
  cd normalSpeed/normalSpeed
  sudo apt-get install libopencv-dev python-opencv
  pip install opencv-python
  python3 setup.py install --user
  cd ..
  ```
  - if pybind can't be found, set pybind11 package path as below in `normalSpeed/CMakeLists.txt`:
  ```shell
  find_package(pybind11 REQUIRED PATHS /home/USERNAME/anaconda3/envs/gaml/lib/python3.6/site-packages/pybind11/share/cmake/pybind11)
  ```
- Install tkinter through ``sudo apt install python3-tk``

- Compile [RandLA-Net](https://github.com/qiqihaer/RandLA-Net-pytorch) operators:
  ```shell
  cd gaml/models/RandLA/
  sh compile_op.sh
  ```
##  Datasets and Checkpoints
Download MCMS dataset and checkpoints from [google drive link](https://drive.google.com/drive/folders/1QUgjZo-xacurUPI82o40HJWuBXPmk6RC?usp=sharing). 
 - copy or link ``MCMS`` folder to ``gaml/datasets/shapenet/MCMS``   
 - copy or link ``checkpoints`` folder to ``gaml/checkpoints``   

##  Training and Evaluation
- Training
```shell
 bash train_shapenet.sh
  ```
  - Visualization
```shell
 bash demo_shapenet_parallel.sh
  ```
  **Note:**  MCMS dataset type, e.g., Toy, PBR or Occ-MCMS can be set in [common_shapenet.py](gaml/common_shapenet.py).
  
## Acknowledgement
This code is based on [FFB6D](https://github.com/ethnhe/FFB6D).  
