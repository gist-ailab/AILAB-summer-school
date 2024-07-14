
# Install Container
```bash
docker login nvcr.io

id : $oauthtoken
pwd : Njc5dHR0b2QwZTh0dTFtNW5ydXI4Y3JtNm46MGVkM2VjODctZTk1Ni00NmNjLTkxNDEtYTdmMjNlNjllMjNj 
```

# Build
```bash
docker build --pull -t \
  registry.ark.svc.ops.openark/library/isaac-sim:2022.2.1-ubuntu22.04_v3 \
  --build-arg ISAACSIM_VERSION=2022.2.1 \
  --build-arg BASE_DIST=ubuntu20.04 \
  --build-arg CUDA_VERSION=11.4.2 \
  --build-arg VULKAN_SDK_VERSION=1.3.224.1 \
  --file Dockerfile.2022.2.1-ubuntu22.04 .
```


```bash
docker push --tls-verify=false registry.ark.svc.ops.openark/library/isaac-sim:2022.2.1-ubuntu22.04_v3
```


# Container Usage
```bash
docker pull --tls-verify=false registry.ark.svc.ops.openark/library/isaac-sim:2022.2.1-ubuntu22.04_v3
```

```bash
podman run -it --entrypoint bash --name isaac-sim --device nvidia.com/gpu=all -e "ACCEPT_EULA=Y" --rm --network=host \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-e DISPLAY \
-v /home:/home \
-v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache/Kit:rw \
-v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
-v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
-v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
-v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
-v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
-v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
-v ~/docker/isaac-sim/documents:/root/Documents:rw \
registry.ark.svc.ops.openark/library/isaac-sim:2022.2.1-ubuntu22.04_v3
```


# In a docker container (Deprecated)
```bash
alias code="code --no-sandbox --user-data-dir=/root"
alias chrome="google-chrome --no-sandbox"

ln -sf /usr/lib64/libcuda.so.1 /usr/lib64/libcuda.so
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64
```
