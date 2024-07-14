# AILAB-summer-school-2024

## Environment Setup

### Setting Step by Step
#### 0. Clone Repository
```
git clone -b 24summer https://github.com/gist-ailab/AILAB-summer-school.git
```

#### 1. Download Isaac Sim
 - Dependency check
    - Ubuntu
      - Recommanded: 20.04 / 22.04
      - Tested on: 20.04
    - NVIDIA Driver version
      - Recommanded: 525.60.11
      - Minimum: 510.73.05
      - Tested on: 510.108.03 / 
 - [Download Omniverse](https://developer.nvidia.com/isaac-sim)
 

#### 2. Python

- Check [Python Environment Installation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/install_python.html#advanced-running-with-anaconda)
 
- move to `Isaac-sim` installed path
   ``` Bash
   version=2023.1.1
   isaac_path=/home/${USER}/.local/share/ov/pkg/isaac-sim-${version}
   cd ${isaac_path}
   ```

- Create env create
   ``` Bash
   conda env create -f environment.yml
   # This process can take a long time.
   conda activate isaac-sim
   ```

- Setup environment variables so that Isaac Sim python packages are located correctly
   ``` Bash
   source setup_conda_env.sh
   ```

- Install requirment pakages
   ``` Bash
   # This procedure takes place within the AILAB-summer-school repository.
   pip install -r requirements.txt
   ```

- Install pycocotools
   ``` Bash
   conda install -c conda-forge pycocotools
   ```

- Install CLIP
   ``` Bash
   pip install git+https://github.com/openai/CLIP.git
   ```

####

#### 3. Checkpoint

- Install gdown
   ``` Bash
   pip install gdown
   # This can be skipped if you have previously installed 'requirements.txt'.
   ```

- Detection model checkpoint download
   ``` Bash
   cd lecture/data
   mkdir -p checkpoint/faster_r-cnn_ckpt
   cd checkpoint/faster_r-cnn_ckpt
   gdown https://drive.google.com/uc?id=16AnvrmyTgm-1xZMIQTmKc4aZVgj76OXt
   ```
   
- Grasp model checkpoint download
   ``` Bash
   cd lecture/data
   mkdir -p checkpoint/contact_grasp_ckpt
   cd checkpoint/contact_grasp_ckpt
   gdown https://drive.google.com/uc?id=16XYFNjSosM7W7DxXUNcI9VNGIPbol6tY
   ```

#### 4. Dataset

- Usd files download
   ``` Bash
   cd lecture/data
   gdown https://drive.google.com/uc?id=1SA9Q6HPGmsNEY4RNGUMHsFq3HtGRoP_1
   unzip scene_generate_usd.zip
   ```

### Docker (2024)
- install Docker on local & set permission
  ``` Bash
  sudo apt install docker.io
  sudo usermod -aG docker ${USER}
  sudo service docker restart
  # To apply permission settings, you must log out and reconnect the logged-in session.
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
  ```login
  docker login -v docker.io
  
  id : birdomi
  pwd : dckr_pat_c3afN9jUpcVUVMqKlWxzFNL8Y_Y
  ```
  ``` Docker
  docker pull docker.io/birdomi/ailab-summer-camp-2024:1.0.2
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
   birdomi/ailab-summer-camp-2024:1.0.2
   ```
- In the docker,some alias are alreadly set.
   ``` Docker
   code: run visual studio code.
   sim: run isaac-sim.
   ```
