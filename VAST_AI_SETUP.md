# 🚀 IDM-VTON Vast.ai Kurulum Rehberi

Bu rehber, IDM-VTON Virtual Try-On sistemini Vast.ai üzerinde kurmak ve API olarak çalıştırmak için hazırlanmıştır.

## 📋 Gereksinimler

### Vast.ai Sunucu Gereksinimleri
- **GPU**: Tesla V100 veya daha üstü (minimum 16GB VRAM)
- **RAM**: En az 32GB
- **Disk**: En az 50GB (model dosyaları için)
- **CUDA**: 11.8 veya üstü

### Önerilen Vast.ai Konfigürasyonu
```bash
# Vast.ai CLI komutu (şablonunuzdan)
vastai create instance <OFFER_ID> \
    --image vastai/pytorch:@vastai-automatic-tag \
    --env '-p 8000:8000 -p 7860:7860' \
    --disk 50 \
    --ssh
```

## 🎯 Kurulum Adımları

### 1. Vast.ai Sunucuya Bağlanma

```bash
# SSH ile bağlanın
ssh root@YOUR_INSTANCE_IP -p YOUR_SSH_PORT
```

### 2. Proje Dosyalarını Yükleme

```bash
# Workspace dizinine geçin
cd /workspace

# Projenizi GitHub'dan klonlayın veya dosyaları upload edin
git clone https://github.com/yisol/IDM-VTON.git
cd IDM-VTON

# API dosyalarını ekleyin (bu rehberdeki dosyaları)
# vast_setup.sh, api_server.py, api_requirements.txt, etc.
```

### 3. Otomatik Kurulum

```bash
# Kurulum scriptini çalıştırın
chmod +x vast_setup.sh
./vast_setup.sh
```

### 4. Manuel Kurulum (alternatif)

Eğer otomatik kurulum çalışmazsa:

```bash
# Conda environment oluşturun
conda env create -f environment.yaml
conda activate idm

# API bağımlılıklarını yükleyin
pip install -r api_requirements.txt

# Model dosyalarını indirin
mkdir -p ckpt/densepose ckpt/humanparsing ckpt/openpose/ckpts

# DensePose modeli
wget -O ckpt/densepose/model_final_162be9.pkl \
    "https://dl.fbaipublicfiles.com/detectron2/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl"

# Human parsing modelleri
wget -O ckpt/humanparsing/parsing_atr.onnx \
    "https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/humanparsing/parsing_atr.onnx"
wget -O ckpt/humanparsing/parsing_lip.onnx \
    "https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/humanparsing/parsing_lip.onnx"

# OpenPose modeli
wget -O ckpt/openpose/ckpts/body_pose_model.pth \
    "https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/openpose/ckpts/body_pose_model.pth"

# IP-Adapter
git clone https://huggingface.co/h94/IP-Adapter
cp -r IP-Adapter/sdxl_models/* ckpt/ip_adapter/
cp -r IP-Adapter/models/image_encoder/* ckpt/image_encoder/
```

## 🌐 Servisleri Başlatma

### Sadece API Server (Önerilen)

```bash
python api_server.py
```

API şu adreste erişilebilir olacak:
- **API Dokumentasyonu**: `http://YOUR_INSTANCE_IP:8000/docs`
- **Health Check**: `http://YOUR_INSTANCE_IP:8000/health`

### Sadece Gradio Interface

```bash
python gradio_demo/app.py
```

Web interface: `http://YOUR_INSTANCE_IP:7860`

### Her İkisi Birden

```bash
# Terminal 1
python gradio_demo/app.py &

# Terminal 2
python api_server.py
```

## 🔧 API Kullanımı

### Python Client Örneği

```python
import requests
import base64

# API endpoint
api_url = "http://YOUR_INSTANCE_IP:8000"

# Dosya upload ile
files = {
    'human_image': open('human.jpg', 'rb'),
    'garment_image': open('garment.jpg', 'rb')
}
data = {
    'garment_description': 'red t-shirt',
    'auto_mask': True,
    'denoise_steps': 30,
    'seed': 42
}

response = requests.post(f"{api_url}/try-on", files=files, data=data)
result = response.json()

if result['success']:
    # Result image'ı kaydet
    with open('result.png', 'wb') as f:
        f.write(base64.b64decode(result['result_image']))
```

### cURL Örneği

```bash
curl -X POST "http://YOUR_INSTANCE_IP:8000/try-on" \
     -F "human_image=@human.jpg" \
     -F "garment_image=@garment.jpg" \
     -F "garment_description=red t-shirt" \
     -F "auto_mask=true" \
     -F "denoise_steps=30" \
     -F "seed=42"
```

## 📊 API Endpoints

### GET `/health`
Sunucu durumunu kontrol eder.

### POST `/try-on`
Dosya upload ile virtual try-on.

**Parametreler:**
- `human_image`: İnsan fotoğrafı (file)
- `garment_image`: Giysi fotoğrafı (file)
- `garment_description`: Giysi açıklaması (string)
- `auto_mask`: Otomatik mask kullan (boolean, default: true)
- `auto_crop`: Otomatik crop kullan (boolean, default: false)
- `denoise_steps`: Denoising adımları (int, 20-40, default: 30)
- `seed`: Random seed (int, optional)

### POST `/try-on-base64`
Base64 encoded images ile virtual try-on.

### POST `/load-models`
Modelleri GPU'ya yükler.

### POST `/unload-models`
Modelleri GPU'dan kaldırır (memory tasarrufu için).

## 🐳 Docker ile Kurulum (Alternatif)

```bash
# Docker image build et
docker build -t idm-vton .

# Sadece API çalıştır
docker run --gpus all -p 8000:8000 -e SERVICE_MODE=api idm-vton

# Gradio interface çalıştır
docker run --gpus all -p 7860:7860 -e SERVICE_MODE=gradio idm-vton

# Her ikisi birden çalıştır
docker run --gpus all -p 8000:8000 -p 7860:7860 -e SERVICE_MODE=both idm-vton
```

### Docker Compose ile

```bash
# Sadece API
docker-compose up idm-vton-api

# Sadece Gradio
docker-compose --profile gradio-only up

# Her ikisi birden
docker-compose --profile both up
```

## 🔥 Performans Optimizasyonu

### GPU Memory Yönetimi

```python
# Modelleri ihtiyaç halinde yükle/kaldır
requests.post(f"{api_url}/load-models")    # GPU'ya yükle
requests.post(f"{api_url}/unload-models")  # CPU'ya taşı
```

### Batch İşleme

```python
# Çoklu istekler için modelleri GPU'da tutun
requests.post(f"{api_url}/load-models")

for image_pair in image_pairs:
    result = requests.post(f"{api_url}/try-on", ...)
    # Process result

# İş bitince GPU'yu temizle
requests.post(f"{api_url}/unload-models")
```

## 🛠️ Troubleshooting

### Yaygın Problemler

**Problem**: CUDA out of memory
```bash
# Çözüm: Modelleri CPU'ya taşı
curl -X POST "http://localhost:8000/unload-models"
```

**Problem**: Model dosyaları eksik
```bash
# Çözüm: Setup scriptini tekrar çalıştır
./vast_setup.sh
```

**Problem**: Port erişim sorunu
```bash
# Vast.ai port konfigürasyonunu kontrol et
# Instance ayarlarından 8000 ve 7860 portlarını açık olduğundan emin ol
```

### Log Kontrol

```bash
# API logs
tail -f api.log

# Gradio logs
tail -f gradio.log

# System logs
journalctl -f
```

## 📈 Monitoring

### GPU Kullanımı

```bash
# GPU status
nvidia-smi

# Sürekli monitoring
watch -n 1 nvidia-smi
```

### API Performansı

```bash
# Health check
curl http://localhost:8000/health

# Detailed status
curl http://localhost:8000/health | jq
```

## 🔒 Güvenlik

### Production için Öneriler

1. **CORS ayarları** - `api_server.py` dosyasında `allow_origins` kısmını düzenleyin
2. **Rate limiting** ekleyin
3. **API key authentication** ekleyin
4. **HTTPS** kullanın

### Firewall

```bash
# Sadece gerekli portları aç
ufw allow 8000  # API
ufw allow 7860  # Gradio
ufw enable
```

## 📞 Destek

### Test Scripti

```bash
# API'yi test et
python test_api.py
```

### Yararlı Komutlar

```bash
# Disk kullanımı
df -h

# Memory kullanımı
free -h

# Process listesi
ps aux | grep python

# Port kontrolü
netstat -tlnp | grep :8000
```

## ✅ Başarı Kontrolü

Kurulum başarılı ise:
1. ✅ `http://YOUR_INSTANCE_IP:8000/health` OK döner
2. ✅ `http://YOUR_INSTANCE_IP:8000/docs` API dokumentasyonunu gösterir
3. ✅ Test istekleri başarılı sonuç döner

---

## 🎉 Tebrikler!

IDM-VTON sisteminiz artık Vast.ai üzerinde çalışıyor ve API olarak kullanıma hazır!

### Sonraki Adımlar
- Kendi uygulamanızdan API'yi kullanmaya başlayın
- Performans optimizasyonları yapın
- İhtiyaçlarınıza göre API'yi genişletin 