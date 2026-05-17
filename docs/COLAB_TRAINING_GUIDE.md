# วิธีเทรนโมเดลบน Google Colab (Pro User)

## 📋 คำอธิบายทีละขั้นตอน:
ดูรายละเอียดทีละ cell ได้ที่: <<ref_file file="D:\Heat-wave-backend\docs\COLAB_STEP_BY_STEP.md" />

## ข้อดีของ Colab Pro:
- ✅ **รันได้ 24 ชั่วโมงต่อ session** (ฟรีแค่ 12 ชั่วโมง)
- ✅ **GPU ดีกว่า:** A100, V100, T4 (ฟรีแค่ T4)
- ✅ **Memory มากกว่า:** เทรนเร็วขึ้น
- ✅ **ไม่หยุดกลางคัน:** ไม่ต้องกลัว disconnect

## เวลาที่ใช้ (Pro):
- **A100 GPU:** 2-3 ชั่วโมง (เร็วที่สุด)
- **V100 GPU:** 3-4 ชั่วโมง (เร็วมาก)
- **T4 GPU:** 4-6 ชั่วโมง (มาตรฐาน)

## สิ่งที่ต้องเตรียมก่อน:

### 1. Upload data ไป Google Drive (สำคัญมาก!)
- เปิด Google Drive (drive.google.com)
- Upload folder `data/` ทั้งหมด (4.1GB) ไป My Drive
- รอจน upload เสร็จ (ประมาณ 10-30 นาที)

### 2. เปิด Google Colab
1. ไปที่: https://colab.research.google.com/
2. File → Open notebook → Upload
3. เลือกไฟล์: `notebooks/train_on_colab.ipynb`

### 3. เลือก GPU ที่ดีที่สุด (Pro)
1. คลิก "Runtime" ด้านบน
2. คลิก "Change runtime type"
3. **Pro เลือกได้:** "A100 GPU" (ดีที่สุด), "V100 GPU" (ดีมาก), หรือ "T4 GPU" (มาตรฐาน)
4. คลิก "Save"

**💡 Tips:** A100 เร็วที่สุดแต่อาจไม่เสมอไป V100 ก็ยังดีมากสำหรับ ML

## วิธีรัน (ง่ายมาก!):

### คลิกปุ่ม ▶️ ทีละ cell จากบนลงล่าง:

**Cell 1:** เปิด Colab (อ่าน instruction)
**Cell 2:** Clone โปรเจกต์จาก GitHub (อัตโนมัติ)
**Cell 3:** เช็ค GPU
**Cell 4:** Install dependencies (อัตโนมัติ)
**Cell 5:** Upload data จาก Google Drive (ต้อง upload ไป Drive ก่อน)
**Cell 6:** Verify setup
**Cell 7:** Test GPU
**Cell 8:** เริ่มเทรน (ใช้เวลา 2-6 ชั่วโมง ขึ้นอยู่กับ GPU)
**Cell 9:** Download models (เสร็จแล้ว)

## เวลาที่ต้องใช้ (Pro):

- **Upload data:** 10-30 นาที
- **Setup:** 5-10 นาที
- **Training:** 2-6 ชั่วโมง (ขึ้นอยู่กับ GPU: A100=2-3h, V100=3-4h, T4=4-6h)
- **Download:** 5-10 นาที

## รวมทั้งหมด: 3-7 ชั่วโมง (เร็วกว่าฟรีมาก!)

## ถ้ามีปัญหา (Pro User):

**Colab หยุดทำงาน?**
- Pro มี limit 24 ชั่วโมง (เกือบไม่เคยถึง)
- ถ้าหยุด รันใหม่ได้เลย - จะทำต่อจากที่หยุด

**ไม่ได้ A100 GPU?**
- A100 มีจำกัดตามเวลา/ภูมิภาค
- V100 ก็ยังดีมากสำหรับ ML
- T4 ก็ยังใช้ได้ดีถ้าอื่นไม่ว่าง

**GPU ไม่ทำงาน?**
- ตรวจสอบว่าเลือก GPU ใน Runtime แล้ว
- Pro มี GPU หลายแบบให้เลือก
- ถ้า GPU ล้มเหลว จะใช้ CPU อัตโนมัติ

**Data upload ล้มเหลว?**
- ลองใช้ Google Drive (วิธีที่เชื่อถือได้ที่สุด)
- Pro มี quota ใน Drive มากกว่า
- ตรวจสอบว่ามีพื้นที่ใน Drive เพียงพอ

## เสร็จแล้วทำอย่างไร:

1. Extract ZIP file ที่ download มา
2. Copy models ไป `app/ml/forecast/models/`
3. Test ด้วย API

## สรุป (Pro User):

✓ ง่ายมาก - แค่คลิกปุ่ม ▶️
✓ เร็วกว่าฟรีมาก - 2-6 ชั่วโมง (vs 5-7 ชั่วโมง)
✓ GPU ดีกว่า - A100, V100, T4
✓ รันได้ 24 ชั่วโมง - เทรนทั้งหมดในครั้งเดียว
✓ เทรนอัตโนมัติทั้งหมด

**เพียงแค่ upload data ไป Google Drive ก่อน แล้วคลิกปุ่ม ▶️ ทีละ cell!**

## เปรียบเทียบ Pro vs ฟรี:

| คุณสมบัติ | ฟรี | Pro |
|------------|------|-----|
| Runtime | 12 ชม. | 24 ชม. |
| GPU | T4 เท่านั้น | A100, V100, T4 |
| Memory | จำกัด | มากกว่า |
| เวลาเทรน | 5-7 ชม. | 3-7 ชม. |
| ราคา | ฟรี | $10/เดือน |

**Pro คุ้มมากสำหรับเทรนโมเดล!**
