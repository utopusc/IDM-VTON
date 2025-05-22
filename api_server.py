from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import io
import base64
import sys
import os
sys.path.append('./')

# IDM-VTON imports
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    AutoTokenizer
)
from diffusers import DDPMScheduler, AutoencoderKL
import numpy as np
from utils_mask import get_mask_location
from torchvision import transforms
import apply_net
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation
from torchvision.transforms.functional import to_pil_image

app = FastAPI(title="IDM-VTON API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
pipe = None
parsing_model = None
openpose_model = None
tensor_transform = None

def pil_to_binary_mask(pil_image, threshold=0):
    np_image = np.array(pil_image)
    grayscale_image = Image.fromarray(np_image).convert("L")
    binary_mask = np.array(grayscale_image) > threshold
    mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    for i in range(binary_mask.shape[0]):
        for j in range(binary_mask.shape[1]):
            if binary_mask[i,j] == True:
                mask[i,j] = 1
    mask = (mask*255).astype(np.uint8)
    output_mask = Image.fromarray(mask)
    return output_mask

def initialize_models():
    global pipe, parsing_model, openpose_model, tensor_transform
    
    print("🚀 Initializing models...")
    
    # Model path - Docker container içinde
    base_path = './models/idm-vton'
    
    print("📦 Loading UNet models...")
    unet = UNet2DConditionModel.from_pretrained(
        base_path,
        subfolder="unet",
        torch_dtype=torch.float16,
    )
    unet.requires_grad_(False)
    
    UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
        base_path,
        subfolder="unet_encoder",
        torch_dtype=torch.float16,
    )
    UNet_Encoder.requires_grad_(False)
    
    print("📦 Loading tokenizers...")
    tokenizer_one = AutoTokenizer.from_pretrained(
        base_path,
        subfolder="tokenizer",
        revision=None,
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        base_path,
        subfolder="tokenizer_2",
        revision=None,
        use_fast=False,
    )
    
    print("📦 Loading text encoders...")
    text_encoder_one = CLIPTextModel.from_pretrained(
        base_path,
        subfolder="text_encoder",
        torch_dtype=torch.float16,
    )
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        base_path,
        subfolder="text_encoder_2",
        torch_dtype=torch.float16,
    )
    
    print("📦 Loading image encoder...")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        base_path,
        subfolder="image_encoder",
        torch_dtype=torch.float16,
    )
    
    print("📦 Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        base_path,
        subfolder="vae",
        torch_dtype=torch.float16,
    )
    
    print("📦 Loading scheduler...")
    noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")
    
    # Freeze models
    image_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    
    print("🔧 Creating pipeline...")
    pipe = TryonPipeline.from_pretrained(
        base_path,
        unet=unet,
        vae=vae,
        feature_extractor=CLIPImageProcessor(),
        text_encoder=text_encoder_one,
        text_encoder_2=text_encoder_two,
        tokenizer=tokenizer_one,
        tokenizer_2=tokenizer_two,
        scheduler=noise_scheduler,
        image_encoder=image_encoder,
        torch_dtype=torch.float16,
    )
    pipe.unet_encoder = UNet_Encoder
    
    print("🔧 Loading parsing and pose models...")
    parsing_model = Parsing(0)
    openpose_model = OpenPose(0)
    
    tensor_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    print("✅ All models loaded successfully!")
    return pipe, parsing_model, openpose_model

def perform_tryon(human_img, garm_img, garment_des, is_checked, is_checked_crop, denoise_steps, seed):
    global pipe, parsing_model, openpose_model, tensor_transform
    
    openpose_model.preprocessor.body_estimation.model.to(device)
    pipe.to(device)
    pipe.unet_encoder.to(device)

    garm_img = garm_img.convert("RGB").resize((768, 1024))
    human_img_orig = human_img.convert("RGB")
    
    if is_checked_crop:
        width, height = human_img_orig.size
        target_width = int(min(width, height * (3 / 4)))
        target_height = int(min(height, width * (4 / 3)))
        left = (width - target_width) / 2
        top = (height - target_height) / 2
        right = (width + target_width) / 2
        bottom = (height + target_height) / 2
        cropped_img = human_img_orig.crop((left, top, right, bottom))
        crop_size = cropped_img.size
        human_img = cropped_img.resize((768, 1024))
    else:
        human_img = human_img_orig.resize((768, 1024))

    if is_checked:
        keypoints = openpose_model(human_img.resize((384, 512)))
        model_parse, _ = parsing_model(human_img.resize((384, 512)))
        mask, mask_gray = get_mask_location('hd', "upper_body", model_parse, keypoints)
        mask = mask.resize((768, 1024))
    else:
        # Manual mask kullanımı için placeholder
        mask = Image.new('L', (768, 1024), 255)
    
    mask_gray = (1 - transforms.ToTensor()(mask)) * tensor_transform(human_img)
    mask_gray = to_pil_image((mask_gray + 1.0) / 2.0)

    human_img_arg = _apply_exif_orientation(human_img.resize((384, 512)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")

    args = apply_net.create_argument_parser().parse_args((
        'show', './configs/densepose_rcnn_R_50_FPN_s1x.yaml', 
        './ckpt/densepose/model_final_162be9.pkl', 'dp_segm', '-v', 
        '--opts', 'MODEL.DEVICE', device
    ))
    pose_img = args.func(args, human_img_arg)
    pose_img = pose_img[:, :, ::-1]
    pose_img = Image.fromarray(pose_img).resize((768, 1024))

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            with torch.inference_mode():
                prompt = "model is wearing " + garment_des
                negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                
                (
                    prompt_embeds,
                    negative_prompt_embeds,
                    pooled_prompt_embeds,
                    negative_pooled_prompt_embeds,
                ) = pipe.encode_prompt(
                    prompt,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=True,
                    negative_prompt=negative_prompt,
                )

                prompt_c = "a photo of " + garment_des
                (
                    prompt_embeds_c,
                    _,
                    _,
                    _,
                ) = pipe.encode_prompt(
                    [prompt_c],
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                    negative_prompt=[negative_prompt],
                )

                pose_img = tensor_transform(pose_img).unsqueeze(0).to(device, torch.float16)
                garm_tensor = tensor_transform(garm_img).unsqueeze(0).to(device, torch.float16)
                generator = torch.Generator(device).manual_seed(seed) if seed is not None else None
                
                images = pipe(
                    prompt_embeds=prompt_embeds.to(device, torch.float16),
                    negative_prompt_embeds=negative_prompt_embeds.to(device, torch.float16),
                    pooled_prompt_embeds=pooled_prompt_embeds.to(device, torch.float16),
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device, torch.float16),
                    num_inference_steps=denoise_steps,
                    generator=generator,
                    strength=1.0,
                    pose_img=pose_img.to(device, torch.float16),
                    text_embeds_cloth=prompt_embeds_c.to(device, torch.float16),
                    cloth=garm_tensor.to(device, torch.float16),
                    mask_image=mask,
                    image=human_img,
                    height=1024,
                    width=768,
                    ip_adapter_image=garm_img.resize((768, 1024)),
                    guidance_scale=2.0,
                )[0]

    if is_checked_crop:
        out_img = images[0].resize(crop_size)
        human_img_orig.paste(out_img, (int(left), int(top)))
        return human_img_orig, mask_gray
    else:
        return images[0], mask_gray

@app.on_event("startup")
async def startup_event():
    global pipe, parsing_model, openpose_model
    pipe, parsing_model, openpose_model = initialize_models()

@app.get("/")
async def root():
    return {"message": "IDM-VTON API is running!", "status": "ok"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "models_loaded": pipe is not None
    }

@app.post("/try-on")
async def virtual_tryon(
    human_image: UploadFile = File(..., description="Human image"),
    garment_image: UploadFile = File(..., description="Garment image"),
    garment_description: str = Form(..., description="Description of the garment"),
    auto_mask: bool = Form(True, description="Use automatic masking"),
    auto_crop: bool = Form(False, description="Use automatic cropping"),
    denoise_steps: int = Form(30, description="Number of denoising steps"),
    seed: int = Form(42, description="Random seed")
):
    try:
        if pipe is None:
            raise HTTPException(status_code=503, detail="Models not loaded yet")
        
        # Load images
        human_img = Image.open(io.BytesIO(await human_image.read()))
        garm_img = Image.open(io.BytesIO(await garment_image.read()))
        
        # Perform virtual try-on
        result_img, mask_img = perform_tryon(
            human_img, garm_img, garment_description,
            auto_mask, auto_crop, denoise_steps, seed
        )
        
        # Convert result to base64
        img_buffer = io.BytesIO()
        result_img.save(img_buffer, format='PNG', quality=95)
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        
        return JSONResponse({
            "success": True,
            "result_image": f"data:image/png;base64,{img_str}",
            "message": "Virtual try-on completed successfully"
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "message": "An error occurred during virtual try-on"
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)