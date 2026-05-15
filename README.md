---
title: MLOps ResNet API
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
---

# MLOps ResNet-50 API

# MLOps ResNet-50 Image Classification API
This project deploys a quantized ResNet-50 model using FastAPI and Docker, hosted on Hugging Face Spaces.

## 🚀 How to use
The API accepts HTTP POST requests with a JPEG/PNG image file.

### cURL Command Example
To test the API, open your terminal and run the following command (replace `test_image.jpg` with your actual image path):

```bash
curl -X POST "[https://chanuwat0201110-mlops-resnet-api.hf.space/predict](https://chanuwat0201110-mlops-resnet-api.hf.space/predict)" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_image.jpg;type=image/jpeg"