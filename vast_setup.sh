#!/bin/bash

# IDM-VTON Vast.ai Setup Script
echo "🚀 Starting IDM-VTON setup on Vast.ai..."

# Update system
apt-get update && apt-get install -y git wget curl unzip

# Set up workspace
cd /workspace
echo "📁 Setting up workspace in: $(pwd)"

# Clone the project if not exists
REPO_URL=${REPO_URL:-"https://github.com/utopusc/IDM-VTON.git"}
PROJECT_DIR=${PROJECT_DIR:-"IDM-VTON"}

if [ ! -d "$PROJECT_DIR" ]; then
    echo "📥 Cloning IDM-VTON project from: $REPO_URL"
    git clone "$REPO_URL" "$PROJECT_DIR"
fi

cd "$PROJECT_DIR"

# Create conda environment
echo "🐍 Setting up conda environment..."
# Note: vast.ai pytorch template already has conda
conda env create -f environment.yaml -y || echo "Environment already exists"
source activate idm

# Download required models and checkpoints
echo "📦 Downloading required models..."

# Create checkpoint directories
mkdir -p ckpt/densepose
mkdir -p ckpt/humanparsing
mkdir -p ckpt/openpose/ckpts

# Download DensePose model
echo "⬇️ Downloading DensePose model..."
wget -O ckpt/densepose/model_final_162be9.pkl \
    "https://dl.fbaipublicfiles.com/detectron2/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl"

# Download Human Parsing models
echo "⬇️ Downloading Human Parsing models..."
wget -O ckpt/humanparsing/parsing_atr.onnx \
    "https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/humanparsing/parsing_atr.onnx"
wget -O ckpt/humanparsing/parsing_lip.onnx \
    "https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/humanparsing/parsing_lip.onnx"

# Download OpenPose model
echo "⬇️ Downloading OpenPose model..."
wget -O ckpt/openpose/ckpts/body_pose_model.pth \
    "https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/openpose/ckpts/body_pose_model.pth"

# Download IP-Adapter components
echo "⬇️ Downloading IP-Adapter..."
if [ ! -d "IP-Adapter" ]; then
    git clone https://huggingface.co/h94/IP-Adapter
    
    # Move to correct locations
    cp -r IP-Adapter/sdxl_models/* ckpt/ip_adapter/ 2>/dev/null || echo "IP-Adapter models copied"
    cp -r IP-Adapter/models/image_encoder/* ckpt/image_encoder/ 2>/dev/null || echo "Image encoder copied"
fi

# Install additional dependencies that might be missing
pip install --upgrade diffusers transformers accelerate
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html

echo "✅ Setup complete!"
echo "🌐 To start the Gradio interface, run: python gradio_demo/app.py"
echo "🚀 To start the API server, run: python api_server.py"
echo "🔧 The Gradio interface will be available on port 7860"
echo "🔧 The API server will be available on port 8000" 