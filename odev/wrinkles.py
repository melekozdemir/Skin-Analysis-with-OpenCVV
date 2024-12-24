import cv2
import dlib
import numpy as np

# dlib'in yüz tespiti için kullandığı model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    r"C:\Users\mlkoz\OneDrive\Desktop\Python\odev\shape_predictor_68_face_landmarks (1).dat")


def detect_redness(face_region):
    """
    Yüz bölgesinde kızarıklık tespiti yapar.
    """
    hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    red_mask = cv2.inRange(hsv, lower_red, upper_red)
    return cv2.bitwise_and(face_region, face_region, mask=red_mask)


def detect_wrinkles(face_region, lower_threshold=50, upper_threshold=150):
    """
    Yüz bölgesinde kırışıklık tespiti yapar (Canny Kenar Algılama).
    """
    gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray_face, lower_threshold, upper_threshold)


def process_frame(frame):
    """
    Bir görüntü üzerinde kenar algılama ve yüz tespiti işlemleri yapar.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Görüntüyü griye çevir

    # Yüz tespiti
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)  # Yüzdeki özellikleri tespit et

        # Yüz bölgesini tespit et
        x1 = landmarks.part(0).x
        y1 = landmarks.part(19).y
        x2 = landmarks.part(16).x
        y2 = landmarks.part(8).y
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)
        face_region = frame[y1:y2, x1:x2]
        if face_region.size == 0:
            continue

        # Kızarıklık tespiti
        red_area = detect_redness(face_region)
        # Kırışıklık tespiti
        wrinkles = detect_wrinkles(face_region)

        # Kırışıklıkları mavi renkte çiz
        frame[y1:y2, x1:x2][wrinkles != 0] = [255, 0, 0]

        # Alın bölgesini seç
        x1_alin = landmarks.part(17).x
        y1_alin = max(0, landmarks.part(19).y - 30)
        x2_alin = landmarks.part(26).x
        y2_alin = landmarks.part(19).y
        x1_alin, y1_alin, x2_alin, y2_alin = max(0, x1_alin), max(0, y1_alin), min(frame.shape[1], x2_alin), min(
            frame.shape[0], y2_alin)
        alin_region = frame[y1_alin:y2_alin, x1_alin:x2_alin]
        if alin_region.size == 0:
            continue

        # Alındaki pürüzleri tespit et
        alin_wrinkles = detect_wrinkles(alin_region, lower_threshold=30, upper_threshold=100)
        # Alındaki pürüzleri mavi çiz
        frame[y1_alin:y2_alin, x1_alin:x2_alin][alin_wrinkles != 0] = [255, 0, 0]

    # Tüm çıktıyı göster
    cv2.imshow("Face with Wrinkles and Redness", frame)


def main():
    """
    Kameradan canlı olarak video alır ve her kareyi işler.
    """
    # Kamerayı başlat
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera açılamadı!")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Görüntü alınamadı!")
            break

        # Her bir kareyi işle
        process_frame(frame)

        # 'q' tuşuna basıldığında çıkış yap
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
