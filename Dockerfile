# 1. เลือกใช้ Base Image แบบ 'slim' ซึ่งเป็นเวอร์ชันตัดไขมันออก ทำให้ขนาดเริ่มต้นเล็กมาก
FROM python:3.12-slim

WORKDIR /app

# 2. ติดตั้งเครื่องมือระดับ OS เท่าที่จำเป็น และ "ลบ Cache ทิ้งทันที" เพื่อลดขนาดไฟล์
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# 3. ติดตั้งไลบรารี Python พร้อมคำสั่ง --no-cache-dir เพื่อไม่ให้ระบบเก็บไฟล์ขยะจากการดาวน์โหลด
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "api.py"]