import onnxruntime as ort
from transformers import AutoImageProcessor
from PIL import Image
import requests
import time
import numpy as np

# 1. เตรียมรูปภาพทดสอบ (ใช้รูปเดิมหรือรูปในเครื่องคุณก็ได้)
image_path = r"D:\AI\ModleTest\Dog\2.png" 
image = Image.open(image_path)

# 2. เตรียม Processor (ใช้ของ Hugging Face เหมือนเดิมเพื่อความง่าย)
processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
inputs = processor(image, return_tensors="np") # แปลงเป็น numpy array (ไม่ใช่ pt แล้ว เพราะ ONNX ใช้ numpy)
pixel_values = inputs["pixel_values"]

# --- ฟังก์ชันสำหรับทดสอบความเร็ว ---
def test_speed(model_path, model_name):
    # โหลดโมเดล ONNX
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    
    # วอร์มอัพ (รันเล่นๆ 2 รอบให้ระบบพร้อม)
    for _ in range(2):
        session.run(None, {input_name: pixel_values})
    
    # จับเวลาจริง
    start_time = time.time()
    outputs = session.run(None, {input_name: pixel_values})
    latency = time.time() - start_time
    
    # ดึงคลาสที่ทำนายได้
    logits = outputs[0]
    predicted_label = logits.argmax(-1).item()
    
    print(f"--- {model_name} ---")
    print(f"เวลาที่ใช้ (Latency): {latency:.4f} วินาที")
    return latency

# --- รันเปรียบเทียบ ---
print("กำลังทดสอบความเร็ว...\n")

# ทดสอบ ONNX ปกติ (FP32)
fp32_latency = test_speed("./resnet50_onnx_model/model.onnx", "ONNX ปกติ (FP32)")

# ทดสอบ ONNX แบบบีบอัด (INT8)
int8_latency = test_speed("./resnet50_onnx_model/model_quantized.onnx", "ONNX แบบบีบอัด (INT8 Quantized)")
