# 🎯 IDM-VTON API Setup - Özet

Bu klasörde, IDM-VTON projesini Vast.ai üzerinde API olarak çalıştırmak için gerekli tüm dosyalar hazırlanmıştır.

## 📁 Oluşturulan Dosyalar

### 🔧 Kurulum Dosyaları
- **`vast_setup.sh`** - Vast.ai sunucusunda otomatik kurulum scripti
- **`api_requirements.txt`** - API için gerekli Python paketleri
- **`VAST_AI_SETUP.md`** - Detaylı kurulum rehberi

### 🚀 API Dosyaları
- **`api_server.py`** - FastAPI server (ana API uygulaması)
- **`test_api.py`** - API test scripti

### 🐳 Docker Dosyaları
- **`Dockerfile`** - Container image tanımı
- **`docker-compose.yml`** - Multi-service konfigürasyonu
- **`docker-entrypoint.sh`** - Container başlangıç scripti

## 🎯 Hızlı Başlangıç

### 1. Vast.ai'da Sunucu Kirala
Verdiğiniz konfigürasyon ile Tesla V100 sunucu kiralayın.

### 2. Dosyaları Yükle
```bash
# SSH ile sunucuya bağlan
ssh root@YOUR_INSTANCE_IP -p YOUR_SSH_PORT

# Workspace'e geç
cd /workspace

# Bu projeyi klonla veya dosyaları upload et
```

### 3. Kurulumu Çalıştır
```bash
chmod +x vast_setup.sh
./vast_setup.sh
```

### 4. API'yi Başlat
```bash
python api_server.py
```

### 5. Test Et
```bash
# Health check
curl http://YOUR_INSTANCE_IP:8000/health

# API dökümantasyonu
# http://YOUR_INSTANCE_IP:8000/docs adresini tarayıcıda aç
```

## 📱 API Kullanımı

### Python Client Örneği
```python
import requests

# Dosya ile istek
files = {
    'human_image': open('human.jpg', 'rb'),
    'garment_image': open('garment.jpg', 'rb')
}
data = {
    'garment_description': 'red t-shirt',
    'auto_mask': True,
    'denoise_steps': 30
}

response = requests.post(
    "http://YOUR_INSTANCE_IP:8000/try-on", 
    files=files, 
    data=data
)
result = response.json()
```

### cURL Örneği
```bash
curl -X POST "http://YOUR_INSTANCE_IP:8000/try-on" \
     -F "human_image=@human.jpg" \
     -F "garment_image=@garment.jpg" \
     -F "garment_description=red t-shirt"
```

## 🔧 Servis Modları

API server 3 farklı modda çalışabilir:

### 1. Sadece API (Önerilen)
```bash
python api_server.py
# Port 8000'de API çalışır
```

### 2. Sadece Gradio
```bash
python gradio_demo/app.py
# Port 7860'da web interface çalışır
```

### 3. Her İkisi Birden
```bash
python gradio_demo/app.py &
python api_server.py
# Hem 8000 hem 7860 portları aktif
```

## 🐳 Docker ile Kullanım

### Tek Container
```bash
# API mode
docker run --gpus all -p 8000:8000 -e SERVICE_MODE=api idm-vton

# Gradio mode
docker run --gpus all -p 7860:7860 -e SERVICE_MODE=gradio idm-vton

# Both modes
docker run --gpus all -p 8000:8000 -p 7860:7860 -e SERVICE_MODE=both idm-vton
```

### Docker Compose
```bash
# Sadece API
docker-compose up idm-vton-api

# Gradio interface
docker-compose --profile gradio-only up

# Her ikisi birden
docker-compose --profile both up
```

## 📊 API Endpoints

| Method | Endpoint | Açıklama |
|--------|----------|----------|
| GET | `/` | Ana sayfa |
| GET | `/health` | Sistem durumu |
| GET | `/docs` | API dökümantasyonu |
| POST | `/try-on` | Virtual try-on (file upload) |
| POST | `/try-on-base64` | Virtual try-on (base64) |
| POST | `/load-models` | Modelleri GPU'ya yükle |
| POST | `/unload-models` | Modelleri CPU'ya taşı |

## ⚡ Performans İpuçları

### GPU Memory Yönetimi
```python
# İşlem öncesi modelleri yükle
requests.post("http://localhost:8000/load-models")

# Batch işlemler yap
for request in requests:
    result = process_request(request)

# İşlem sonrası temizle
requests.post("http://localhost:8000/unload-models")
```

### Optimal Parametreler
- **denoise_steps**: 20-30 (hız için), 30-40 (kalite için)
- **auto_mask**: True (çoğu durumda)
- **auto_crop**: False (özel durumlar için True)

## 🛠️ Troubleshooting

### Yaygın Sorunlar

**CUDA out of memory**
```bash
curl -X POST "http://localhost:8000/unload-models"
```

**Model dosyaları eksik**
```bash
./vast_setup.sh
```

**API yanıt vermiyor**
```bash
# Port kontrolü
netstat -tlnp | grep :8000

# Process kontrolü
ps aux | grep api_server
```

### Log Kontrolü
```bash
# API logları
tail -f api.log

# GPU kullanımı
nvidia-smi

# Sistem durumu
curl http://localhost:8000/health
```

## 🔗 Başka Araçlar Ekleme

API server çalıştıktan sonra diğer araçları da ekleyebilirsiniz:

### ComfyUI Ekleme
```bash
cd /workspace
git clone https://github.com/comfyanonymous/ComfyUI.git
# Port 8188'de çalıştır
```

### Automatic1111 Ekleme
```bash
cd /workspace
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
# Port 7861'de çalıştır
```

### Fooocus Ekleme
```bash
cd /workspace
git clone https://github.com/lllyasviel/Fooocus.git
# Port 7862'de çalıştır
```

## 📞 Destek

### Test Scripti
```bash
python test_api.py
```

### Monitoring
```bash
# Sistem durumu
curl http://localhost:8000/health | jq

# GPU monitoring
watch -n 1 nvidia-smi
```

---

## ✅ Başarı Kriterleri

Kurulum başarılı ise:
1. ✅ Health check OK döner
2. ✅ API dökümantasyonu erişilebilir
3. ✅ Test istekleri çalışır
4. ✅ GPU memory kullanımı görünür

## 🎉 Sonuç

Bu dosyalar ile IDM-VTON sisteminizi Vast.ai üzerinde:
- ✅ Hızlıca kurabilirsiniz
- ✅ API olarak kullanabilirsiniz  
- ✅ Diğer araçlarla entegre edebilirsiniz
- ✅ Ölçeklendirebilirsiniz

**Happy coding! 🚀** 