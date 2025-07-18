import cv2


camera_url = "rtsp://admin:123456@192.168.1.100:554/stream1"


cap = cv2.VideoCapture(camera_url)

# ตรวจสอบว่าเปิดกล้องได้หรือไม่
if not cap.isOpened():
    print("ไม่สามารถเชื่อมต่อกับ IP Camera ได้ โปรดตรวจสอบ URL หรือการตั้งค่ากล้อง")
    exit()

print("เชื่อมต่อกับ IP Camera สำเร็จ กด 'q' เพื่อออก")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # (Optional) Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # ตัวอย่างแนวคิดด้วย Traditional Methods
    # การใช้ Sobel operator เพื่อหาขอบในแนวแกน X (ป้ายทะเบียนมีขอบแนวตั้งเด่นชัด)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_x = cv2.convertScaleAbs(sobel_x)

    # Thresholding
    _, binary_image = cv2.threshold(sobel_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Closing Operation to connect broken edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    license_plate_roi = None
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h
        area = cv2.contourArea(cnt)

        # กรอง Contours ที่เป็นไปได้สำหรับป้ายทะเบียนไทย (ปรับค่าเหล่านี้)
        if 2.0 < aspect_ratio < 5.0 and 1000 < area < 20000: # ค่า area อาจต้องปรับตามระยะกล้อง
            # อาจจะต้องมี heuristic เพิ่มเติม เช่น ความเข้มของ pixel ภายใน contour
            # หรือการตรวจสอบความกว้าง/สูงขั้นต่ำ

            # Crop ROI
            license_plate_roi = gray[y:y+h, x:x+w]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            break # พบป้ายแรกแล้วออก (อาจจะต้องปรับหากมีหลายป้าย)

    if license_plate_roi is not None:
        # ทำ OCR ต่อไป
        pass
    # แสดงผลเฟรม
    cv2.imshow('IP Camera Feed', frame)

    # กด 'q' เพื่อออกจากลูป
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปล่อยทรัพยากร
cap.release()
cv2.destroyAllWindows()