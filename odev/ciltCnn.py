import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam


def load_skin_type_model():
    """
    Cilt tipi sınıflandırması için eğitimli modeli yükler.
    Burada bir önceden eğitilmiş CNN modeli kullanılabilir.
    """
    # compile=False kullanarak modeli yükleyin
    model = load_model(r'C:\Users\mlkoz\OneDrive\Desktop\Python\odev\modell.h5', compile=False)

    # Modeli yeniden derleyin
    optimizer = Adam(learning_rate=0.005)  # lr yerine learning_rate kullanıyoruz
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def predict_skin_type(model, face_image):
    """
    Modeli kullanarak cilt tipi tahmini yapar.
    """
    # Yüzü uygun boyutlara getir
    face_image = cv2.resize(face_image, (224, 224))  # Modelin giriş boyutuna göre
    face_image = np.expand_dims(face_image, axis=0)  # Batch boyutu ekleyin
    face_image = face_image / 255.0  # Normalizasyon

    # Modeli kullanarak tahmin yap
    prediction = model.predict(face_image)
    skin_type = np.argmax(prediction, axis=1)  # Tahmin edilen sınıf

    # Tahmin sonucunu kontrol et
    if len(skin_type) == 0:
        return "Bilinmiyor"

    # Cilt tipi etiketini döndür
    skin_types = ['Yağlı Cilt', 'Kuru Cilt', 'Normal Cilt', 'Karma Cilt']
    if skin_type[0] >= len(skin_types):
        return "Bilinmiyor"
    return skin_types[skin_type[0]]


def detect_faces(gray, face_cascade):
    """
    Gri tonlamalı görüntüde yüzleri tespit eder.
    """
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    return faces


def segment_skin(frame):
    """
    HSV renk uzayında cilt tonlarını maskeleme işlemi.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([0, 48, 80], dtype=np.uint8)
    upper_hsv = np.array([20, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
    upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
    skin_mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

    skin_mask = cv2.bitwise_and(skin_mask, skin_mask_ycrcb)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    skin_mask = cv2.erode(skin_mask, kernel, iterations=2)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
    skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)

    return skin_mask


def enhance_contrast(gray):
    """
    Gri tonlamalı görüntünün kontrastını artırır.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return enhanced


def process_frame(frame, face_cascade, model):
    """
    Yüz tespiti ve cilt tipi tahminini yapar.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray, face_cascade)

    for (x, y, w, h) in faces:
        face_roi = frame[y:y + h, x:x + w]

        # Cilt tipi tahminini yap
        skin_type = predict_skin_type(model, face_roi)

        # Cilt tipi metnini ekle
        cv2.putText(frame, f"Cilt Tipi: {skin_type}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Cilt maskesini oluştur ve segmentasyonu göster
        skin_mask = segment_skin(face_roi)
        skin = cv2.bitwise_and(face_roi, face_roi, mask=skin_mask)

        # Gri tonlama ve kontrast artırma
        skin_gray = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)
        skin_gray = enhance_contrast(skin_gray)

        # Sonuçları orijinal görüntüye entegre et
        frame[y:y + h, x:x + w] = cv2.addWeighted(face_roi, 0.7, skin, 0.3, 0)

    return frame


def main():
    """
    Kameradan canlı görüntü alır ve yüz üzerindeki cilt pürüzlerini algılar.
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        print("Yüz algılama modeli yüklenemedi!")
        return

    # Modeli yükle
    model = load_skin_type_model()

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

        # Kareyi işle
        processed_frame = process_frame(frame, face_cascade, model)

        # Görüntüyü göster
        cv2.imshow("Cilt Tipi ve Segmentasyon", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()