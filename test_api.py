from fastapi.testclient import TestClient
from api import app
import io
from PIL import Image

client = TestClient(app)

def test_predict_valid_image():
    """ทดสอบอัปโหลดรูปภาพจำลอง ระบบต้องคืนค่า 200"""
    # สร้างรูปภาพจำลอง (สีดำ 100x100) ขึ้นมาใน Memory ไม่ต้องอ่านจากไดรฟ์
    image = Image.new('RGB', (100, 100))
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    response = client.post(
        "/predict",
        files={"file": ("test_image.jpg", img_byte_arr, "image/jpeg")}
    )
    
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert "prediction" in response.json()["data"]

def test_predict_invalid_file_type():
    """ทดสอบอัปโหลดไฟล์ text ระบบต้องดักจับ Error และคืนค่า 400"""
    fake_text_bytes = b"This is a text file, not an image."
    
    response = client.post(
        "/predict",
        files={"file": ("test.txt", fake_text_bytes, "text/plain")}
    )
    
    assert response.status_code == 400
    assert "File must be an image" in response.json()["detail"]