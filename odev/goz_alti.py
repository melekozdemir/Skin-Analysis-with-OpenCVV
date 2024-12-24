
import cv2
import numpy as np


def detect_and_analyze(camera_index=0):
    """
    Kamera ile yüz ve göz altı analizi yapar.
    """
    # Kamerayı başlat
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise ValueError("Kamera açılamadı! Lütfen cihazınıza bağlı bir kamera olduğundan emin olun.")

    # Haarcascade modellerini yükle
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    print("Kamera açık, analiz ediliyor... Çıkmak için 'q' tuşuna basın.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kamera görüntüsü alınamadı, çıkılıyor...")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Yüzleri algıla
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            # Yüz bölgesini ayır
            face_region_gray = gray[y:y + h, x:x + w]
            face_region_color = frame[y:y + h, x:x + w]

            # Gözleri algıla (yüzün üst yarısında arama yap)
            eye_search_region = face_region_gray[:h // 2, :]
            eyes = eye_cascade.detectMultiScale(eye_search_region, scaleFactor=1.1, minNeighbors=5)

            eye_data = []
            for (ex, ey, ew, eh) in eyes:
                # Göz altı bölgesini belirle
                under_eye_top = max(ey + eh - 10, 0)
                under_eye_bottom = min(under_eye_top + int(0.4 * eh), face_region_color.shape[0])
                under_eye_region = face_region_color[under_eye_top:under_eye_bottom, ex:ex + ew]

                if (under_eye_region.size
                        == 0):
                    continue

                # Analiz et ve sonuçları kaydet
                analysis = analyze_under_eye_region(under_eye_region)
                eye_data.append(analysis)

                # Çerçeve çiz ve analiz sonucunu görüntüye ekle
                cv2.rectangle(face_region_color, (ex, under_eye_top), (ex + ew, under_eye_bottom), analysis['color'], 2)

            # Eğer iki göz analiz edildiyse ortalama değerlere göre genel öneri oluştur
            if len(eye_data) == 2:
                recommendation = generate_recommendation(eye_data)
                cv2.putText(frame, recommendation, (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Görüntüyü göster
        cv2.imshow("Göz Altı Analizi", frame)

        # Çıkış için 'q' tuşuna bas
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def analyze_under_eye_region(under_eye_region):
    """
    Göz altı bölgesini analiz eder.
    """
    b, g, r = cv2.split(under_eye_region)
    avg_blue = np.mean(b)
    avg_red_green = (np.mean(r) + np.mean(g)) / 2

    gray_eye = cv2.cvtColor(under_eye_region, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_eye, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if avg_blue > 120:
        return {"problem": "Morluk", "color": (255, 0, 0), "details": "Mavi ton yüksek"}
    elif avg_red_green > 130:
        return {"problem": "Kahvelik", "color": (42, 42, 165), "details": "Kahverengi ton yüksek"}
    elif len(contours) > 0:
        return {"problem": "Göz Altı Torbaları", "color": (255, 255, 255),
                "details": f"{len(contours)} kontur tespit edildi"}
    else:
        return {"problem": "Normal", "color": (0, 255, 0), "details": "Herhangi bir problem tespit edilmedi"}


def generate_recommendation(eye_data):
    """
    İki gözün analiz sonuçlarına göre genel bir öneri oluşturur.
    """
    problems = [data['problem'] for data in eye_data]
    if problems.count("Morluk") == 2:
        return "Goz altlariniz koyu. Daha fazla dinlenin!"
    elif problems.count("Kahvelik") == 2:
        return "Goz altlariniz kahverengi. Göz kremleri kullanmayi deneyin."
    elif problems.count("Göz Altı Torbaları") == 2:
        return "Goz altlariniz şişkin. Soguk kompres önerilir."
    else:
        return "Goz altlariniz genel olarak iyi durumda."


# Kamera üzerinden analiz yap
if __name__ == "__main__":
    detect_and_analyze()
