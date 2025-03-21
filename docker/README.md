## 🐳 Docker Setup for StarVector

To simplify the setup process and avoid dependency issues, you can use a Docker container to run **StarVector** in a self-contained environment. The Dockerfile and this guide are located in the `docker/` subfolder of the repository.

---

### 🛠️ Build the Docker Image

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/joanrod/star-vector.git
   cd star-vector/docker
   ```

2. **Build the Image**:

   ```bash
   docker build -t starvector:latest .
   ```

---

### 🚀 Run the Docker Container

```bash
docker run -it \
  --gpus all \
  -p 8888:8888 \
  -v $(pwd)/..:/workspace \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env HUGGING_FACE_HUB_TOKEN=<your_huggingface_token> \
  --name starvector \
  starvector:latest
```

**Options explained:**

- `--gpus all`: Enables GPU acceleration (requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))
- `-p 8888:8888`: Exposes Jupyter Notebook to your host machine
- `-v $(pwd)/..:/workspace`: Mounts the project root directory into the container
- `-v ~/.cache/huggingface:/root/.cache/huggingface`: Shares Hugging Face cache for offline usage and faster loading
- `--env HUGGING_FACE_HUB_TOKEN=...`: Sets an environment variable with your token for accessing gated models
- `--name starvector`: Assigns a name to the container

---

### 🔑 Hugging Face Token

**StarVector models depend on** the Hugging Face model `bigcode/starcoderbase-1b`, which is a gated model.

To access it, you need a **Hugging Face token** with the following permission:

> ✅ Read access to contents of all public gated repos you can access

You can generate it here: https://huggingface.co/settings/tokens

---

### 🌐 Access Jupyter Notebook

After the container starts, you'll see a URL like:

```
http://127.0.0.1:8888/?token=<your_token_here>
```

Copy and open it in your browser to start using **StarVector** in a notebook environment.

---