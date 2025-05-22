#!/bin/bash

echo "🚀 Starting IDM-VTON Container..."

# Check if models are downloaded
if [ ! -f "ckpt/densepose/model_final_162be9.pkl" ]; then
    echo "📦 Downloading required models..."
    
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
fi

# Check if IP-Adapter is set up
if [ ! -d "IP-Adapter" ]; then
    echo "⬇️ Downloading IP-Adapter..."
    git clone https://huggingface.co/h94/IP-Adapter
    
    # Copy to correct locations
    cp -r IP-Adapter/sdxl_models/* ckpt/ip_adapter/ 2>/dev/null || echo "IP-Adapter models copied"
    cp -r IP-Adapter/models/image_encoder/* ckpt/image_encoder/ 2>/dev/null || echo "Image encoder copied"
fi

echo "✅ Model setup complete!"

# Start services based on environment variable
SERVICE_MODE=${SERVICE_MODE:-"api"}

case $SERVICE_MODE in
    "gradio")
        echo "🌐 Starting Gradio interface on port 7860..."
        python3 gradio_demo/app.py
        ;;
    "api")
        echo "🚀 Starting FastAPI server on port 8000..."
        python3 api_server.py
        ;;
    "both")
        echo "🚀 Starting both Gradio and API servers..."
        python3 gradio_demo/app.py &
        python3 api_server.py
        ;;
    *)
        echo "🤖 Starting interactive shell..."
        exec bash
        ;;
esac 