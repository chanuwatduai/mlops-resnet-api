import os
from optimum.onnxruntime import ORTModelForImageClassification
from transformers import AutoImageProcessor

model_id = "microsoft/resnet-50"
save_directory = "./resnet50_onnx_model" # โฟลเดอร์สำหรับเก็บโมเดล ONNX

print(f"กำลังโหลดและแปลงโมเดล {model_id} เป็น ONNX ด้วย Optimum...")

# 1. โหลดและแปลงโมเดลอัตโนมัติ (ใส่ export=True คือคีย์เวิร์ดสำคัญ)
model = ORTModelForImageClassification.from_pretrained(model_id, export=True)
processor = AutoImageProcessor.from_pretrained(model_id)

# 2. บันทึกโมเดลและ Processor ลงในโฟลเดอร์
model.save_pretrained(save_directory)
processor.save_pretrained(save_directory)

print(f"\n✅ แปลงไฟล์สำเร็จ! โมเดลถูกบันทึกไว้ที่โฟลเดอร์: {save_directory}")

# 3. เช็คขนาดไฟล์ ONNX ที่อยู่ข้างในโฟลเดอร์
onnx_file_path = os.path.join(save_directory, "model.onnx")
if os.path.exists(onnx_file_path):
    onnx_size_mb = os.path.getsize(onnx_file_path) / (1024 * 1024)
    print(f"ขนาดไฟล์ ONNX (Model Size): {onnx_size_mb:.2f} MB")
else:
    print("ไม่พบไฟล์ model.onnx ในโฟลเดอร์")