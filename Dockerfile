FROM nvidia/cuda:11.8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Sistem güncellemeleri ve gerekli paketler
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgoogle-glog0v5 \
    libgflags2.2 \
    libprotobuf23 \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Python ve pip güncelleme
RUN python3.10 -m pip install --upgrade pip setuptools wheel

# Çalışma dizini
WORKDIR /app

# Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# PyTorch ve CUDA uyumluluğu
RUN pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118

# Detectron2 kurulumu
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Proje dosyalarını kopyala
COPY . .

# Gerekli dizinleri oluştur
RUN mkdir -p ckpt/densepose ckpt/humanparsing ckpt/openpose/ckpts ckpt/ip_adapter models uploads outputs

# Port expose
EXPOSE 8000

# Başlangıç komutu
CMD ["python", "api_server.py"]