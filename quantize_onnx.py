import os
from onnxruntime.quantization import quantize_dynamic, QuantType

# 1. ระบุชื่อไฟล์ต้นทาง (ไฟล์ที่เราเพิ่งได้มา 97.43 MB)
# ต้องชี้ไปที่ไฟล์ model.onnx ที่อยู่ในโฟลเดอร์ที่เราเซฟไว้
model_input_path = "./resnet50_onnx_model/model.onnx"

# 2. ระบุชื่อไฟล์ปลายทาง (ไฟล์ที่จะถูกบีบอัดแล้ว)
model_output_path = "./resnet50_onnx_model/model_quantized.onnx"

print(f"กำลังเริ่มกระบวนการ Dynamic Quantization...")
print(f"อ่านไฟล์จาก: {model_input_path}")

# 3. สั่งทำ Dynamic Quantization
quantize_dynamic(
    model_input=model_input_path,        # ไฟล์ต้นฉบับ
    model_output=model_output_path,      # ไฟล์ผลลัพธ์
    weight_type=QuantType.QUInt8         # แปลงน้ำหนักให้เป็นแบบ 8-bit Integer (ขนาดจะเล็กลงมาก)
)

print(f"\n✅ ทำ Quantization สำเร็จ! บันทึกไฟล์ที่: {model_output_path}")

# 4. เปรียบเทียบขนาดไฟล์
original_size = os.path.getsize(model_input_path) / (1024 * 1024)
quantized_size = os.path.getsize(model_output_path) / (1024 * 1024)

print("-" * 30)
print(f"ขนาดก่อนบีบอัด (ONNX FP32):  {original_size:.2f} MB")
print(f"ขนาดหลังบีบอัด (ONNX INT8): {quantized_size:.2f} MB")
print("-" * 30)