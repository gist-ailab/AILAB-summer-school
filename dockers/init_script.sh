# Docker Pull Container
docker pull --tls-verify=false registry.ark.svc.ops.openark/library/isaac-sim:2022.2.1-ubuntu22.04_v3


# Alias for Docker Terminal Execution
echo 'alias sim="docker exec -it isaac-sim bash"' >> ~/.zshrc
source ~/.zshrc

# Run container
podman run -dit --entrypoint bash --name isaac-sim -e "ACCEPT_EULA=Y" --rm --network=host \
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


# Git Clone Lecture Files
cd /home/user/Desktop
rm -rf AILAB-isaac-sim-pick-place
git clone https://github.com/gist-ailab/AILAB-isaac-sim-pick-place.git --recurse-submodules --remote-submodules


# Run Examples
sim /isaac-sim/isaac-sim.sh --allow-root
