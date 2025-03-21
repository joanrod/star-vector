FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# Install necessary dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    vim \
    build-essential \
    libcairo2 \
    cuda-compiler-12-4 \
    libaio-dev \
    && rm -rf /var/lib/apt/lists/*


# Install Jupyter Notebook, clone and install star-vector and dependencies
RUN pip install --upgrade pip && pip install jupyter deepspeed \
    && git clone https://github.com/joanrod/star-vector.git /tmp/star-vector \
    && pip install /tmp/star-vector \
    && rm -rf /tmp/star-vector

# Cleanup unneeded packages
RUN apt-get purge -y --auto-remove \
    git \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Expose Jupyter Notebook port
EXPOSE 8888

WORKDIR /workspace

# Launch Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
