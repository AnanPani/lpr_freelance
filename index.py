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
        print("ไม่สามารถรับเฟรมจากกล้องได้ อาจมีการตัดการเชื่อมต่อ")
        break

    # แสดงผลเฟรม
    cv2.imshow('IP Camera Feed', frame)

    # กด 'q' เพื่อออกจากลูป
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปล่อยทรัพยากร
cap.release()
cv2.destroyAllWindows()