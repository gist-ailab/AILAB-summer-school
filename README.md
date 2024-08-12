# AILAB-summer-school-2024

## Environment Setup

 - Below procedure is Workstation Setup and instruction to setting Step by Step
 - If you want to use already setup env please follow [docker setup](#docker-2024) instruction

#### 1. Download Isaac Sim
 - Notice: *Following lectures are built on the version of Isaac Sim 2023.1.1.*
 - Dependency check
    - Ubuntu
      - Lecture Tested on: 20.04
      - Isaac Sim Recommanded : 20.04 / 22.04
    - NVIDIA Driver version
      - Lecture Tested on: 510.108.03
      - Isaac Sim Recommanded: 525.60.11
      - Isaac Sim Minimum: 510.73.05
 - [Download Omniverse](https://developer.nvidia.com/isaac-sim)
 

#### 2. Clone Repository
- move to `Isaac-sim` installed path

   ```Bash
   version=2023.1.1
   isaac_path=/home/${USER}/.local/share/ov/pkg/isaac-sim-${version}
   cd ${isaac_path}
   ```

- clone repository
  
   ```Bash
   git clone https://github.com/gist-ailab/AILAB-summer-school.git --recurse-submodules --remote-submodules
   ```

#### 3. Python Setup
 - Notice: *Following lectures are built on the version of Python 3.10 & Anaconda environment*
    - Replaceable with python-venv and miniconda
 - Check [Python Environment Installation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/install_python.html#advanced-running-with-anaconda)
 
 - Create conda env
    - This procedure run inside `isaac sim` path(ex. .../isaac-sim-2023.1.1$)
    - This process can take a long time.
   ```Bash
   conda env create -f environment.yml 
   conda activate isaac-sim
   ```

- Setup environment variables so that Isaac Sim python packages are located correctly
   ```Bash
   source setup_conda_env.sh
   ```

 - Install requirment pakages
    - This procedure takes place within the `AILAB-summer-school` repository.
   ```Bash
   cd AILAB-summer-school
   pip install -r requirements.txt
   ```

 - Install pycocotools
   ```Bash
   conda install -c conda-forge pycocotools
   ```

 - Install CLIP
   ```Bash
   pip install git+https://github.com/openai/CLIP.git
   ```

#### 3. Download Checkpoints and Assets

 - Detection model checkpoint download
   ```Bash
   cd lecture/data
   mkdir -p checkpoint/faster_r-cnn_ckpt
   cd checkpoint/faster_r-cnn_ckpt
   gdown https://drive.google.com/uc?id=16AnvrmyTgm-1xZMIQTmKc4aZVgj76OXt
   ```
   
 - Grasp model checkpoint download
   ```Bash
   cd lecture/data
   mkdir -p checkpoint/contact_grasp_ckpt
   cd checkpoint/contact_grasp_ckpt
   gdown https://drive.google.com/uc?id=16XYFNjSosM7W7DxXUNcI9VNGIPbol6tY
   ```

 - Asset files download
   ```Bash
   cd lecture/data
   gdown https://drive.google.com/uc?id=1SA9Q6HPGmsNEY4RNGUMHsFq3HtGRoP_1
   unzip scene_generate_usd.zip
   ```

 - Detection dataset (PennFudanPed Dataset) download
   ```Bash
   cd lecture/data
   gdown https://drive.google.com/uc?id=15EbsaKLnkhHxGHYejORDTsS9rThELJMR
   unzip PennFudanPed.zip
   ```

## Docker (2024)
 - install Docker on local & set permission
   ``` Bash
   sudo apt install docker.io
   sudo usermod -aG docker ${USER}
   sudo service docker restart
   # To apply permission settings, you must log out and reconnect the logged-in session.
   ```

 - Install the nvidia Docker Container Toolkit to Use the gpu Option in Docker ([official instruction](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html))
   ``` bash
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
   && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
   sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
   sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   ```
   ``` bash
   sudo apt-get update
   ```
   ``` bash
   sudo apt-get install -y nvidia-container-toolkit
   ```
   ``` bash
   sudo systemctl restart docker
   ```

 - Docker cache store for local
   ```
   mkdir -p ~/docker/isaac-sim/cache/kit
   mkdir -p ~/docker/isaac-sim/cache/ov
   mkdir -p ~/docker/isaac-sim/cache/pip
   mkdir -p ~/docker/isaac-sim/cache/glcache
   mkdir -p ~/docker/isaac-sim/cache/computecache
   ```
  
 - Pull docker image
   ``` Docker
   docker login
   
   id : birdomi
   pwd : dckr_pat_c3afN9jUpcVUVMqKlWxzFNL8Y_Y
   ```
   ``` Docker
   docker pull docker.io/birdomi/ailab-summer-camp-2024:1.0.4
   ```
   
 - Start the container
   ``` Docker
   docker run -it --entrypoint bash --name isaac-sim -e "ACCEPT_EULA=Y" --gpus all --rm --network=host \
   -e DISPLAY \
   -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
   -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
   -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
   -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
   -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
   birdomi/ailab-summer-camp-2024:1.0.4
   ```

 - In the docker,some alias are alreadly set.
   ``` Docker
   code: run visual studio code.
   sim: run isaac-sim.
   ```
