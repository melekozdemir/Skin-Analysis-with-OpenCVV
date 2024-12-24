import cv2
import numpy as np

def detect_skin_type(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    if len(faces) == 0:
        print("Yüz algılanamadı.")
        return None

    for (x, y, w, h) in faces:
        face_roi = image[y:y+h, x:x+w]

        # Yüzü dört bölgeye ayırıyoruz: alın, burun, yanaklar ve çene
        height, width, _ = face_roi.shape

        forehead = face_roi[:height // 4, :]
        nose = face_roi[height // 4:height // 2, width // 3:2 * width // 3]
        cheeks = np.hstack((face_roi[height // 4:3 * height // 4, :width // 3],
                            face_roi[height // 4:3 * height // 4, 2 * width // 3:]))
        chin = face_roi[3 * height // 4:, :]

        # Parlaklık ölçümü için gri tonlama ve ortalama parlaklık hesaplama
        def get_brightness(region):
            gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            return np.mean(gray_region)

        brightness_forehead = get_brightness(forehead)
        brightness_nose = get_brightness(nose)
        brightness_cheeks = get_brightness(cheeks)
        brightness_chin = get_brightness(chin)

        # Parlaklık eşik değerleri (kendi kameranızda kalibre edilebilir)
        threshold = 100
        is_forehead_bright = brightness_forehead > threshold
        is_nose_bright = brightness_nose > threshold
        is_cheeks_bright = brightness_cheeks > threshold
        is_chin_bright = brightness_chin > threshold

        # Cilt tipi belirleme
        if is_forehead_bright and is_nose_bright and is_cheeks_bright:
            skin_type = "Yagli Cilt"
        elif is_forehead_bright and is_nose_bright and not is_cheeks_bright:
            skin_type = "Karma Cilt"
        elif is_cheeks_bright and not (is_forehead_bright or is_nose_bright):
            skin_type = "Normal Cilt"
        else:
            skin_type = "Kuru Cilt"

        return skin_type

    return None

# Kameradan görüntü alımı
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kamera açılamadı.")
        break

    skin_type = detect_skin_type(frame)
    if skin_type:
        cv2.putText(frame, f"Cilt Tipi: {skin_type}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Cilt Analizi", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()