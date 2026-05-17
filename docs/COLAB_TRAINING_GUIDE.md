# วิธีเทรนโมเดลบน Google Colab (Pro User)

## 📋 คำอธิบายทีละขั้นตอน

## ข้อดีของ Colab Pro

- ✅ **รันได้ 24 ชั่วโมงต่อ session** (ฟรีแค่ 12 ชั่วโมง)
- ✅ **GPU ดีกว่า:** A100, V100, T4 (ฟรีแค่ T4)
- ✅ **Memory มากกว่า:** เทรนเร็วขึ้น
- ✅ **ไม่หยุดกลางคัน:** ไม่ต้องกลัว disconnect

## เวลาที่ใช้ (Pro)

- **A100 GPU:** 2-3 ชั่วโมง (เร็วที่สุด)
- **V100 GPU:** 3-4 ชั่วโมง (เร็วมาก)
- **T4 GPU:** 4-6 ชั่วโมง (มาตรฐาน)

## สิ่งที่ต้องเตรียมก่อน

### 1. เลือกวิธีโหลด data (4 วิธี)

**วิธีที่ 1: Zip จาก Drive (เร็วที่สุด, 5-10 นาที)** ✨ แนะนำ

- Upload folder `data/` ทั้งหมด (4.1GB) ไป Google Drive My Drive
- รัน Cell 4a ใน notebook เพื่อสร้าง `data.zip` (ครั้งเดียว, 5-10 นาที)
- Cell 4 จะใช้ FASTEST PATH (copy zip → extract)
- **Speedup:** 4-8× เร็วกว่า copy folder

**วิธีที่ 2: Tar จาก Drive (เร็ว, 7-13 นาที)**

- Upload folder `data/` ทั้งหมด (4.1GB) ไป Google Drive My Drive
- สร้าง `data.tar.gz` บน Drive (ครั้งเดียว, 5-10 นาที)
- Cell 4 จะใช้ FAST PATH (copy tar → extract)
- **Speedup:** 3-5× เร็วกว่า copy folder

**วิธีที่ 3: Copy folder จาก Drive (ช้า, 15-40 นาที)**

- Upload folder `data/` ทั้งหมด (4.1GB) ไป Google Drive My Drive
- Cell 4 จะใช้ SLOW PATH (copy folder ผ่าน FUSE)
- **เหมาะสำหรับ:** ครั้งแรกที่ไม่มี zip/tar

**วิธีที่ 4: Ingest NASA POWER (no auth, 5-15 นาที)**

- ไม่ต้อง upload data ไป Drive
- รัน Cell 4b เพื่อ ingest จาก NASA POWER API โดยตรง
- **ข้อดี:** ไม่ต้อง auth, เร็ว
- **ข้อเสีย:** ความแม่นยำต่ำกว่า ERA5

**💡 แนะนำ:** ใช้วิธีที่ 1 (Zip) ถ้ามี data บน Drive แล้ว

### 2. เปิด Google Colab

1. ไปที่: <https://colab.research.google.com/>
2. File → Open notebook → Upload
3. เลือกไฟล์: `notebooks/train_on_colab.ipynb`

### 3. เลือก GPU ที่ดีที่สุด (Pro)

1. คลิก "Runtime" ด้านบน
2. คลิก "Change runtime type"
3. **Pro เลือกได้:** "A100 GPU" (ดีที่สุด), "V100 GPU" (ดีมาก), หรือ "T4 GPU" (มาตรฐาน)
4. คลิก "Save"

**💡 Tips:** A100 เร็วที่สุดแต่อาจไม่เสมอไป V100 ก็ยังดีมากสำหรับ ML

## วิธีรัน (ง่ายมาก!)

### คลิกปุ่ม ▶️ ทีละ cell จากบนลงล่าง

**Cell 1:** Clone โปรเจกต์จาก GitHub (อัตโนมัติ)
**Cell 2:** เช็ค GPU
**Cell 3:** Install dependencies (อัตโนมัติ)
**Cell 4a (OPTIONAL):** สร้าง zip บน Drive (run ครั้งเดียว, 5-10 นาที)
**Cell 4:** Upload data จาก Google Drive (FASTEST: zip, FAST: tar, SLOW: folder)
**Cell 4b (ALTERNATIVE):** Ingest NASA POWER (no auth, 5-15 นาที)
**Cell 5:** Verify setup
**Cell 6:** Test GPU
**Cell 7:** เริ่มเทรน (ใช้เวลา 2-6 ชั่วโมง ขึ้นอยู่กับ GPU)
**Cell 8:** Download models (เสร็จแล้ว)

**💡 Tips:**
- เลือกใช้ Cell 4, 4a, หรือ 4b อย่างใดอย่างหนึ่งเท่านั้น
- ถ้ามี data บน Drive แล้ว → ใช้ Cell 4a (สร้าง zip) + Cell 4 (copy zip)
- ถ้ามี data บน Drive แต่ไม่มี zip/tar → ใช้ Cell 4 (copy folder)
- ถ้าไม่มี data บน Drive → ใช้ Cell 4b (ingest NASA POWER)

## เวลาที่ต้องใช้ (Pro)

**วิธีที่ 1 (Zip - เร็วที่สุด):**
- **Upload data:** 10-30 นาที (ครั้งแรก)
- **สร้าง zip:** 5-10 นาที (ครั้งเดียว)
- **Copy data:** 5-10 นาที (zip → extract)
- **Setup:** 5-10 นาที
- **Training:** 2-6 ชั่วโมง (ขึ้นอยู่กับ GPU: A100=2-3h, V100=3-4h, T4=4-6h)
- **Download:** 5-10 นาที
- **รวมครั้งแรก:** 3-7 ชั่วโมง
- **รวมครั้งต่อไป:** 2.5-6.5 ชั่วโมง (zip อยู่แล้ว)

**วิธีที่ 2 (Tar):**
- **Upload data:** 10-30 นาที (ครั้งแรก)
- **สร้าง tar:** 5-10 นาที (ครั้งเดียว)
- **Copy data:** 7-13 นาที (tar → extract)
- **Setup:** 5-10 นาที
- **Training:** 2-6 ชั่วโมง
- **Download:** 5-10 นาที
- **รวมครั้งแรก:** 3-7 ชั่วโมง
- **รวมครั้งต่อไป:** 2.5-6.5 ชั่วโมง (tar อยู่แล้ว)

**วิธีที่ 3 (Copy folder):**
- **Upload data:** 10-30 นาที
- **Copy data:** 15-40 นาที (ช้ากว่า zip/tar)
- **Setup:** 5-10 นาที
- **Training:** 2-6 ชั่วโมง
- **Download:** 5-10 นาที
- **รวมทั้งหมด:** 3.5-8 ชั่วโมง

**วิธีที่ 4 (NASA POWER):**
- **Upload data:** ไม่ต้อง
- **Ingest data:** 5-15 นาที (no auth)
- **Setup:** 5-10 นาที
- **Training:** 2-6 ชั่วโมง
- **Download:** 5-10 นาที
- **รวมทั้งหมด:** 2.5-7 ชั่วโมง

## ถ้ามีปัญหา (Pro User)

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

## เสร็จแล้วทำอย่างไร

1. Extract ZIP file ที่ download มา
2. Copy models ไป `app/models/forecast_v3/`
3. Test ด้วย API

## สรุป (Pro User)

✓ ง่ายมาก - แค่คลิกปุ่ม ▶️
✓ เร็วกว่าฟรีมาก - 2-7 ชั่วโมง (vs 5-7 ชั่วโมง)
✓ GPU ดีกว่า - A100, V100, T4
✓ รันได้ 24 ชั่วโมง - เทรนทั้งหมดในครั้งเดียว
✓ เทรนอัตโนมัติทั้งหมด
✓ **4 วิธีโหลด data:** Zip (เร็วสุด), Tar (เร็ว), Copy folder (ช้า), NASA POWER (no auth)

**เพียงแค่ upload data ไป Google Drive ก่อน แล้วคลิกปุ่ม ▶️ ทีละ cell!**

## เปรียบเทียบ Pro vs ฟรี

| คุณสมบัติ | ฟรี | Pro |
| ------------ | ------ | ----- |
| Runtime | 12 ชม. | 24 ชม. |
| GPU | T4 เท่านั้น | A100, V100, T4 |
| Memory | จำกัด | มากกว่า |
| เวลาเทรน | 5-7 ชม. | 3-7 ชม. |
| ราคา | ฟรี | $10/เดือน |

**Pro คุ้มมากสำหรับเทรนโมเดล!**
