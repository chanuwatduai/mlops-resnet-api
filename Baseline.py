import torch
import time
from transformers import AutoImageProcessor, ResNetForImageClassification
from PIL import Image

model_name = "microsoft/resnet-50"
processor = AutoImageProcessor.from_pretrained(model_name)
model = ResNetForImageClassification.from_pretrained(model_name)

# 1. ระบุตำแหน่งไฟล์ในเครื่องของคุณ (ใส่ r ไว้ข้างหน้าเพื่อให้ Python อ่าน Path ของ Windows ได้ถูกต้อง)
# อย่าลืมเปลี่ยนชื่อไฟล์ 'dog_01.jpg' เป็นชื่อไฟล์รูปที่มีอยู่จริงในโฟลเดอร์ของคุณนะครับ
image_path = r"D:\AI\ModleTest\Dog\2.png" 

# 2. โหลดรูปภาพด้วย PIL
try:
    image = Image.open(image_path).convert("RGB") # แปลงเป็น RGB เผื่อบางรูปเป็นขาวดำหรือมี Alpha channel
except FileNotFoundError:
    print(f"หารูปไม่เจอ ลองเช็คชื่อไฟล์และ Path อีกครั้ง: {image_path}")
    exit()

# 3. นำเข้าโมเดล
inputs = processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_label = logits.argmax(-1).item()
predicted_class = model.config.id2label[predicted_label]

print(f"ผลการทำนายรูปจากโฟลเดอร์: {predicted_class}")