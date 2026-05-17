# วิธีเทรนโมเดลบน Google Colab (คำอธิบายทีละขั้นตอน)

## ก่อนเริ่ม:
1. Upload folder `data/` (4.1GB) ไป Google Drive My Drive ก่อน
2. เปิด https://colab.research.google.com/
3. เลือก GPU: Runtime → Change runtime type → A100/V100/T4 GPU (Pro) หรือ T4 GPU (ฟรี)
4. Upload notebook: `notebooks/train_on_colab.ipynb`

## วิธีรัน (คลิก ▶️ ทีละ cell จากบนลงล่าง):

### Cell 1: Clone project from GitHub
- **ทำอะไร:** Download โค้ดจาก GitHub มา Colab
- **เวลา:** 1-2 นาที
- **ผลลัพธ์:** ✓ Project cloned successfully

### Cell 2: Check GPU
- **ทำอะไร:** เช็คว่า GPU พร้อมใช้งานไหม
- **เวลา:** 10 วินาที
- **ผลลัพธ์:** แสดงข้อมูล GPU (Tesla T4/V100/A100)

### Cell 3: Install dependencies
- **ทำอะไร:** Install libraries ที่จำเป็น
- **เวลา:** 2-3 นาที
- **ผลลัพธ์:** Successfully installed

### Cell 4: Upload data from Google Drive
- **ทำอะไร:** Copy data จาก Google Drive มา Colab
- **เวลา:** 5-10 นาที
- **ผลลัพธ์:** ✓ Data uploaded successfully
- **⚠️ สำคัญ:** ต้อง upload data ไป Google Drive ก่อนรัน cell นี้

### Cell 5: Verify setup
- **ทำอะไร:** เช็คว่าไฟล์ครบถ้วนไหม
- **เวลา:** 10 วินาที
- **ผลลัพธ์:** แสดง project structure

### Cell 6: Test GPU training
- **ทำอะไร:** ทดสอบว่า GPU training ทำงานได้ไหม
- **เวลา:** 30 วินาที
- **ผลลัพธ์:** ✓ GPU training works! หรือ Will use CPU (ถ้า GPU ไม่ทำงาน)

### Cell 7: Train all models with auto-tuning
- **ทำอะไร:** เทรนโมเดลทั้งหมด 80 โมเดล
- **เวลา:**
  - A100 GPU: 2-3 ชั่วโมง (Pro)
  - V100 GPU: 3-4 ชั่วโมง (Pro)
  - T4 GPU: 4-6 ชั่วโมง (Pro/ฟรี)
  - CPU: 8-12 ชั่วโมง (fallback)
- **ผลลัพธ์:** Training progress แสดงทีละ model
- **💡 สามารถปล่อยให้รันต่อได้ ไม่ต้องดูตลอด**

### Cell 8: Check results
- **ทำอะไร:** เช็คสถานะโมเดลทั้งหมด
- **เวลา:** 10 วินาที
- **ผลลัพธ์:** แสดงตาราง model status ทั้งหมด

### Cell 9: Download trained models
- **ทำอะไร:** Download โมเดลที่เทรนเสร็จแล้ว
- **เวลา:** 5-10 นาที
- **ผลลัพธ์:** trained_models.zip download ไปยัง Downloads folder

### Cell 10: Save to Google Drive
- **ทำอะไร:** Backup โมเดลไป Google Drive
- **เวลา:** 2-3 นาที
- **ผลลัพธ์:** ✓ Models saved to Google Drive

## เวลารวม:
- **Setup:** 10-15 นาที
- **Training:** 2-6 ชั่วโมง (ขึ้นอยู่กับ GPU)
- **Download:** 5-10 นาที
- **รวม:** 3-7 ชั่วโมง

## หลังเทรนเสร็จ:
1. Extract `trained_models.zip`
2. Copy models ไป `app/ml/forecast/models/`
3. Test ด้วย API

## ถ้ามีปัญหา:
- **Training หยุด:** รัน cell 7 ใหม่ - จะทำต่อจากที่หยุด
- **GPU ไม่ทำงาน:** จะใช้ CPU อัตโนมัติ (ช้ากว่าแต่ใช้ได้)
- **Data upload ล้มเหลว:** ตรวจสอบว่า upload data ไป Google Drive แล้ว

## Pro vs ฟรี:
| คุณสมบัติ | ฟรี | Pro |
|------------|------|-----|
| Runtime | 12 ชม. | 24 ชม. |
| GPU | T4 เท่านั้น | A100, V100, T4 |
| เวลาเทรน | 5-7 ชม. | 3-7 ชม. |
| ราคา | ฟรี | $10/เดือน |
