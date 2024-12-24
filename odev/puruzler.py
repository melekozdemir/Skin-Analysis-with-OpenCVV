import cv2
import numpy as np

def detect_faces(gray, face_cascade):
    """
    Gri tonlamalı görüntüde yüzleri tespit eder.
    """
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    return faces

def segment_skin(frame):
    """
    HSV renk uzayında cilt tonlarını maskelemek için kullanılır.
    Daha hassas cilt segmentasyonu için YCrCb renk uzayı da kullanılabilir.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Cilt tonlarını kapsayan HSV aralığı (gerekirse ayarlanabilir)
    lower_hsv = np.array([0, 48, 80], dtype=np.uint8)
    upper_hsv = np.array([20, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # YCrCb renk uzayı ile ek cilt segmentasyonu
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
    upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
    skin_mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

    # İki maske arasında AND işlemi uygulayarak daha hassas bir maske elde ederiz
    skin_mask = cv2.bitwise_and(skin_mask, skin_mask_ycrcb)

    # Gürültüyü azaltmak için morfolojik işlemler
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    skin_mask = cv2.erode(skin_mask, kernel, iterations=2)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
    skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)

    return skin_mask

def enhance_contrast(gray):
    """
    Gri tonlamalı görüntünün kontrastını artırmak için CLAHE kullanılır.
    Bu, parlaklık farklarını dengelemeye yardımcı olur.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    return enhanced

def detect_skin_roughness(skin_gray):
    """
    Cilt gri tonlamalı görüntüsünde pürüzleri algılar.
    """
    # Gaussian Blur ile gürültüyü azaltma
    blurred = cv2.GaussianBlur(skin_gray, (5, 5), 0)

    # Kenar algılama (Laplacian yöntemi)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    laplacian_abs = np.absolute(laplacian)
    laplacian_uint8 = np.uint8(laplacian_abs)

    # Eşikleme ile pürüzleri belirginleştirme
    _, roughness = cv2.threshold(laplacian_uint8, 10, 255, cv2.THRESH_BINARY)

    return roughness

def process_frame(frame, face_cascade):
    """
    Bir kare üzerinde yüz tespiti, cilt segmentasyonu ve pürüz algılama yapar.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray, face_cascade)

    for (x, y, w, h) in faces:
        # Çerçeve boyutunu artırmak için ofset
        offset = 20  # Çerçeveyi her yönden 20 piksel büyüt
        x_new = max(0, x - offset)
        y_new = max(0, y - offset)
        w_new = min(frame.shape[1] - x_new, w + 2 * offset)
        h_new = min(frame.shape[0] - y_new, h + 2 * offset)

        # Büyütülmüş çerçeveye göre yüz bölgesini al
        face_roi = frame[y_new:y_new+h_new, x_new:x_new+w_new]

        # Cilt maskesini oluştur
        skin_mask = segment_skin(face_roi)

        # Cilt bölgesini maskeler
        skin = cv2.bitwise_and(face_roi, face_roi, mask=skin_mask)

        # Gri tonlamaya çevir
        skin_gray = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)

        # Kontrastı artır
        skin_gray = enhance_contrast(skin_gray)

        # Pürüzleri algıla
        roughness = detect_skin_roughness(skin_gray)

        # Pürüz maskesini renkli görüntüye dönüştür
        roughness_color = cv2.cvtColor(roughness, cv2.COLOR_GRAY2BGR)

        # Orijinal yüz bölgesine pürüz maskesini ekle
        frame[y_new:y_new+h_new, x_new:x_new+w_new] = cv2.addWeighted(face_roi, 0.7, roughness_color, 0.3, 0)

    return frame


def main():
    """
    Kameradan canlı görüntü alır ve yüz üzerindeki cilt pürüzlerini algılar.
    """
    # Yüz algılama için Haar Cascade yükleme
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        print("Yüz algılama modeli yüklenemedi!")
        return

    # Kamerayı başlat
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera açılamadı!")
        return

    print("Kamera açık. Çıkmak için 'q' tuşuna basın.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Görüntü alınamadı!")
            break

        # İşlenmiş kareyi al
        processed_frame = process_frame(frame, face_cascade)

        # Orijinal ve işlenmiş görüntüyü göster
        cv2.imshow("Orijinal Görüntü", frame)
        cv2.imshow("Cilt Pürüz Algılama", processed_frame)

        # 'q' tuşuna basıldığında çıkış yap
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Kaynakları serbest bırak ve pencereleri kapat
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
        main()
