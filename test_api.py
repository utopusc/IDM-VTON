#!/usr/bin/env python3
"""
IDM-VTON API Test Script
Test the FastAPI server with sample images
"""

import requests
import base64
import io
import os
from PIL import Image
import json
import time

class IDMVTONAPIClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        
    def health_check(self):
        """Check if the API server is running"""
        try:
            response = requests.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Health check failed: {e}")
            return None
    
    def image_to_base64(self, image_path):
        """Convert image file to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    
    def base64_to_image(self, base64_string):
        """Convert base64 string back to PIL Image"""
        image_data = base64.b64decode(base64_string)
        return Image.open(io.BytesIO(image_data))
    
    def try_on_with_files(self, human_image_path, garment_image_path, garment_description, **kwargs):
        """Try-on using file uploads"""
        url = f"{self.base_url}/try-on"
        
        # Prepare files
        files = {
            'human_image': open(human_image_path, 'rb'),
            'garment_image': open(garment_image_path, 'rb')
        }
        
        # Prepare form data
        data = {
            'garment_description': garment_description,
            'auto_mask': kwargs.get('auto_mask', True),
            'auto_crop': kwargs.get('auto_crop', False),
            'denoise_steps': kwargs.get('denoise_steps', 30),
            'seed': kwargs.get('seed', 42)
        }
        
        try:
            print(f"🚀 Sending try-on request...")
            print(f"Human image: {human_image_path}")
            print(f"Garment image: {garment_image_path}")
            print(f"Description: {garment_description}")
            
            start_time = time.time()
            response = requests.post(url, files=files, data=data, timeout=300)
            end_time = time.time()
            
            response.raise_for_status()
            result = response.json()
            
            print(f"✅ Request completed in {end_time - start_time:.2f} seconds")
            
            return result
            
        except Exception as e:
            print(f"❌ Try-on request failed: {e}")
            return None
        finally:
            # Close file handles
            for f in files.values():
                f.close()
    
    def try_on_with_base64(self, human_image_path, garment_image_path, garment_description, **kwargs):
        """Try-on using base64 encoded images"""
        url = f"{self.base_url}/try-on-base64"
        
        # Convert images to base64
        human_b64 = self.image_to_base64(human_image_path)
        garment_b64 = self.image_to_base64(garment_image_path)
        
        # Prepare form data
        data = {
            'human_image_b64': human_b64,
            'garment_image_b64': garment_b64,
            'garment_description': garment_description,
            'auto_mask': kwargs.get('auto_mask', True),
            'auto_crop': kwargs.get('auto_crop', False),
            'denoise_steps': kwargs.get('denoise_steps', 30),
            'seed': kwargs.get('seed', 42)
        }
        
        try:
            print(f"🚀 Sending try-on request (base64)...")
            print(f"Description: {garment_description}")
            
            start_time = time.time()
            response = requests.post(url, data=data, timeout=300)
            end_time = time.time()
            
            response.raise_for_status()
            result = response.json()
            
            print(f"✅ Request completed in {end_time - start_time:.2f} seconds")
            
            return result
            
        except Exception as e:
            print(f"❌ Try-on request failed: {e}")
            return None
    
    def save_results(self, result, output_dir="outputs"):
        """Save the result images to files"""
        if not result or not result.get('success'):
            print("❌ No valid result to save")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save result image
        if 'result_image' in result:
            result_img = self.base64_to_image(result['result_image'])
            result_path = os.path.join(output_dir, 'result.png')
            result_img.save(result_path)
            print(f"💾 Result image saved to: {result_path}")
        
        # Save mask image
        if 'mask_image' in result:
            mask_img = self.base64_to_image(result['mask_image'])
            mask_path = os.path.join(output_dir, 'mask.png')
            mask_img.save(mask_path)
            print(f"💾 Mask image saved to: {mask_path}")
        
        # Save parameters
        params_path = os.path.join(output_dir, 'parameters.json')
        with open(params_path, 'w') as f:
            json.dump(result.get('parameters', {}), f, indent=2)
        print(f"💾 Parameters saved to: {params_path}")

def main():
    # Initialize client
    client = IDMVTONAPIClient()
    
    # Health check
    print("🔍 Checking API health...")
    health = client.health_check()
    if health:
        print(f"✅ API is healthy: {health}")
    else:
        print("❌ API is not responding")
        return
    
    # Example usage
    # You need to provide actual image paths
    human_image = "path/to/human_image.jpg"
    garment_image = "path/to/garment_image.jpg"
    garment_description = "red t-shirt"
    
    # Check if example images exist
    if not os.path.exists(human_image) or not os.path.exists(garment_image):
        print("⚠️  Example images not found. Please update the paths in the script.")
        print("Expected files:")
        print(f"  - {human_image}")
        print(f"  - {garment_image}")
        return
    
    # Test file upload method
    print("\n📤 Testing file upload method...")
    result = client.try_on_with_files(
        human_image_path=human_image,
        garment_image_path=garment_image,
        garment_description=garment_description,
        denoise_steps=30,
        seed=42
    )
    
    if result:
        client.save_results(result, "outputs/file_upload")
    
    # Test base64 method
    print("\n📤 Testing base64 method...")
    result = client.try_on_with_base64(
        human_image_path=human_image,
        garment_image_path=garment_image,
        garment_description=garment_description,
        denoise_steps=30,
        seed=42
    )
    
    if result:
        client.save_results(result, "outputs/base64")

if __name__ == "__main__":
    main() 