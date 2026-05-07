from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
import onnxruntime as ort
from transformers import AutoImageProcessor, AutoConfig
from PIL import Image
import io
import asyncio
from concurrent.futures import ProcessPoolExecutor

app = FastAPI(title="MLOps Image Classification API")

executor = ProcessPoolExecutor(max_workers=2)

# --- ย้ายตัวแปรมารอไว้ แต่ยังไม่ต้องโหลด ---
# เพื่อให้แต่ละ Process ที่ถูกสร้างขึ้นมา มีตัวแปรเหล่านี้เป็นของตัวเอง
_processor = None
_session = None
_config = None

def init_worker():
    """
    ฟังก์ชันนี้จะทำงาน 1 ครั้ง เมื่อแต่ละ Process ใน Pool ถูกสร้างขึ้น
    เราจะโหลดโมเดลไว้ในนี้ เพื่อให้โมเดลถูกโหลดแยกกันในแต่ละ Process
    และไม่ต้องเสี่ยงกับการส่งข้าม (Pickle)
    """
    global _processor, _session, _config
    model_name = "microsoft/resnet-50"
    _processor = AutoImageProcessor.from_pretrained(model_name)
    _config = AutoConfig.from_pretrained(model_name)
    _session = ort.InferenceSession("./resnet50_onnx_model/model_quantized.onnx")

def process_image(image_bytes: bytes) -> dict:
    """ฟังก์ชันทำนายผล ที่จะถูกรันใน Worker Process"""
    # โหลดโมเดลในครั้งแรกที่ถูกเรียก (ถ้ายังไม่ได้โหลด)
    if _session is None:
        init_worker()

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return {"error": "Invalid or corrupted image file."}

    inputs = _processor(image, return_tensors="np")
    pixel_values = inputs["pixel_values"]
    input_name = _session.get_inputs()[0].name
    
    outputs = _session.run(None, {input_name: pixel_values})
    predicted_label = outputs[0].argmax(-1).item()
    
    predicted_class = _config.id2label[predicted_label]
    return {"prediction": predicted_class}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    
    file_bytes = await file.read()
    
    if len(file_bytes) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size exceeds 5MB limit.")
    
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(executor, process_image, file_bytes)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return {
        "status": "success",
        "filename": file.filename,
        "data": result
    }

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)