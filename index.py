import cv2
import pytesseract
import numpy as np
import re

IP_CAMERA_URL = "rtsp://admin:123456@192.168.1.100:554/stream1"

# --- ฟังก์ชันช่วยเหลือ ---

def preprocess_plate_roi(plate_image):
    """
    ปรับปรุงภาพ ROI ของป้ายทะเบียนให้เหมาะกับการทำ OCR
    """
    # ทำให้เป็น Grayscale (ถ้ายังไม่เป็น)
    if len(plate_image.shape) == 3:
        plate_image = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    
    # Adaptive Thresholding: ปรับค่า threshold ตามแต่ละพื้นที่ของภาพ
    # เพื่อให้ตัวอักษรเป็นสีดำ และพื้นหลังเป็นสีขาว
    processed_image = cv2.adaptiveThreshold(
        plate_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # ขยายภาพเล็กน้อยเพื่อช่วยให้ OCR อ่านได้ดีขึ้น (ถ้าจำเป็น)
    # processed_image = cv2.resize(processed_image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    return processed_image

def clean_and_format_thai_license_plate(text):
    """
    ทำความสะอาดและจัดรูปแบบข้อความที่ได้จาก OCR ให้เป็นป้ายทะเบียนไทย
    """
    # ลบอักขระที่ไม่ใช่ตัวอักษรไทย, อังกฤษ หรือตัวเลข
    cleaned_text = re.sub(r'[^ก-ฮA-Za-z0-9]', '', text).strip()
    
    # รูปแบบทั่วไปของป้ายทะเบียนไทย: อักษร 1-2 ตัว ตามด้วยตัวเลข 1-4 ตัว
    # อาจจะต้องปรับ regex ให้ซับซ้อนขึ้นสำหรับป้ายบางประเภท
    match = re.search(r'([ก-ฮA-Za-z]{1,2})(\d{1,4})', cleaned_text)
    
    if match:
        prefix = match.group(1).upper() # แปลงเป็นตัวพิมพ์ใหญ่
        numbers = match.group(2)
        
        # ตรวจสอบความยาวของตัวเลข (ป้ายทะเบียนทั่วไปมี 1-4 หลัก)
        if 1 <= len(numbers) <= 4:
            return f"{prefix} {numbers}"
    
    return "ไม่พบรูปแบบป้ายทะเบียนที่ชัดเจน"


# --- การเชื่อมต่อและประมวลผลวิดีโอ ---

def main():
    cap = cv2.VideoCapture(IP_CAMERA_URL)

    if not cap.isOpened():
        print(f"!!! ไม่สามารถเชื่อมต่อกับ IP Camera ได้: {IP_CAMERA_URL}")
        print("โปรดตรวจสอบ URL, การตั้งค่ากล้อง, หรือการเชื่อมต่อเครือข่าย")
        return

    print(f"เชื่อมต่อกับ IP Camera สำเร็จ: {IP_CAMERA_URL}")
    print("กำลังประมวลผลวิดีโอ... กด 'q' เพื่อออก")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("ไม่สามารถรับเฟรมจากกล้องได้ อาจมีการตัดการเชื่อมต่อ หรือวิดีโอจบลง")
            break

        # ลดขนาดเฟรมเพื่อเพิ่มความเร็วในการประมวลผล (ถ้าจำเป็น)
        # frame = cv2.resize(frame, (640, 480)) 

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. การตรวจจับขอบในแนวตั้ง (ป้ายทะเบียนมักมีเส้นแนวตั้งชัดเจน)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_x = np.absolute(sobel_x)
        sobel_x = np.uint8(sobel_x)
        
        # 2. ทำ Threshold เพื่อแปลงเป็นภาพขาวดำ
        # cv2.THRESH_OTSU จะหาค่า threshold ที่ดีที่สุดโดยอัตโนมัติ
        _, binary_image = cv2.threshold(sobel_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 3. Morphology Close เพื่อเชื่อมต่อช่องว่างเล็กๆ ในขอบ
        # Kernel ขนาดใหญ่ในแนวนอนเพื่อเชื่อมเส้นที่ขาดบนป้าย
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5)) 
        closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
        
        # 4. หา Contours ที่เป็นไปได้ว่าเป็นป้ายทะเบียน
        contours, _ = cv2.findContours(closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_license_plate_text = "กำลังค้นหา..."
        license_plate_found = False

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            area = cv2.contourArea(cnt)
            
            # กรอง Contours ตามขนาดและอัตราส่วน (ต้องปรับให้เหมาะสมกับระยะและมุมกล้อง)
            # ค่าเหล่านี้เป็นค่าประมาณที่ใช้กับป้ายทะเบียนรถไทย
            if (1.5 < aspect_ratio < 5.0 and # อัตราส่วนกว้าง/ยาวของป้ายทะเบียนไทยมักอยู่ระหว่าง 2:1 ถึง 5:1
                w > 80 and h > 20 and # ขนาดขั้นต่ำของป้ายที่ตรวจจับได้
                area > 1500):         # พื้นที่ขั้นต่ำของ Contour
                
                # ตัดภาพส่วนที่เป็นป้ายทะเบียน
                # เพิ่มขอบเล็กน้อยเพื่อจับขอบป้ายได้ครบถ้วน
                # clip_x, clip_y, clip_w, clip_h = x-5, y-5, w+10, h+10
                # clip_x = max(0, clip_x)
                # clip_y = max(0, clip_y)
                # clip_w = min(frame.shape[1] - clip_x, clip_w)
                # clip_h = min(frame.shape[0] - clip_y, clip_h)

                # ปรับ ROI ให้ครอบคลุมมากขึ้น (อาจช่วย OCR)
                padding = 5 
                lp_x = max(0, x - padding)
                lp_y = max(0, y - padding)
                lp_w = min(frame.shape[1] - lp_x, w + 2 * padding)
                lp_h = min(frame.shape[0] - lp_y, h + 2 * padding)


                license_plate_roi = gray[lp_y : lp_y + lp_h, lp_x : lp_x + lp_w]

                if license_plate_roi.shape[0] > 0 and license_plate_roi.shape[1] > 0:
                    # ประมวลผลภาพป้ายทะเบียนก่อนส่งให้ OCR
                    processed_lp_roi = preprocess_plate_roi(license_plate_roi)
                    
                    # ทำ OCR
                    # lang='tha+eng' คือให้ Tesseract พยายามอ่านทั้งภาษาไทยและอังกฤษ
                    # --psm 7: Assume the image is a single text line. (ดีสำหรับป้ายทะเบียน)
                    ocr_text = pytesseract.image_to_string(processed_lp_roi, lang='tha+eng', config='--psm 7')
                    
                    # ทำความสะอาดและจัดรูปแบบ
                    formatted_lp = clean_and_format_thai_license_plate(ocr_text)
                    
                    if formatted_lp != "ไม่พบรูปแบบป้ายทะเบียนที่ชัดเจน":
                        detected_license_plate_text = formatted_lp
                        license_plate_found = True
                        
                        # วาดสี่เหลี่ยมรอบป้ายทะเบียนที่ตรวจพบ
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        
                        # แสดงผลลัพธ์ OCR บนภาพ
                        cv2.putText(frame, detected_license_plate_text, (x, y - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # แสดงภาพป้ายทะเบียนที่ถูกประมวลผล (สำหรับ Debug)
                        cv2.imshow('Processed License Plate ROI', processed_lp_roi)
                        break # พบป้ายทะเบียนแล้ว ออกจากลูป contours

        # แสดงผลเฟรมวิดีโอ
        cv2.imshow('IP Camera Feed - LPR', frame)
        
        # แสดงผลลัพธ์สุดท้าย
        if not license_plate_found:
            cv2.putText(frame, "No Plate Detected", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


        # กด 'q' เพื่อออกจากลูป
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ปล่อยทรัพยากร
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()