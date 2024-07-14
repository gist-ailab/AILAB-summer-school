# Install Container
```bash
docker login nvcr.io

id : $oauthtoken
pwd : Njc5dHR0b2QwZTh0dTFtNW5ydXI4Y3JtNm46MGVkM2VjODctZTk1Ni00NmNjLTkxNDEtYTdmMjNlNjllMjNj 
```

# INSTALL .config
```
git init
git remote add -f origin https://github.com/ulagbulag/openark-desktop-template.git
git config core.sparseCheckout true
echo ".config/" >> .git/info/sparse-checkout
git pull origin master
```

# Build
```bash
docker build --pull -t \
  registry.ark.svc.ops.openark/library/isaac-sim:2023.1.1-ubuntu20.04 \
  --build-arg ISAACSIM_VERSION=2023.1.1 \
  --build-arg BASE_DIST=ubuntu20.04 \
  --build-arg CUDA_VERSION=11.4.2 \
  --build-arg VULKAN_SDK_VERSION=1.3.224.1 \
  --file Dockerfile.2023.1.1-ubuntu20.04 .
```


```bash
docker push --tls-verify=false registry.ark.svc.ops.openark/library/isaac-sim:2023.1.1-ubuntu20.04
```


# Container Usage
```bash
docker login -v docker.io

id : birdomi
pwd : dckr_pat_c3afN9jUpcVUVMqKlWxzFNL8Y_Y
```

```bash
docker pull docker.io/birdomi/ailab-summer-camp-2024:v1
```

```bash
docker run -it --entrypoint bash --name isaac-sim -e "ACCEPT_EULA=Y" --gpus all --rm --network=host \
-e DISPLAY \
-v /home:/home \
birdomi/ailab-summer-camp-2024:v1
```


# In a docker container (Deprecated)
```bash
alias code="code --no-sandbox --user-data-dir=/root"
alias chrome="google-chrome --no-sandbox"

ln -sf /usr/lib64/libcuda.so.1 /usr/lib64/libcuda.so
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64
```
