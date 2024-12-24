# OpenCV ile ciltteki pürüzleri algılama
import cv2
import numpy as np

# Görüntüyü yükle
image_path = "../odev/akne1.png"
image = cv2.imread(image_path)
image=cv2.resize(image,(400,400))

if image is None:
    print("Görüntü yüklenemedi!")
    exit()

# Grayscale'e dönüştür
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Canny kenar tespiti ile pürüzleri bul
edges = cv2.Canny(gray, 50, 150)

# Gaussian Blur ile pürüzleri yumuşat
blurred = cv2.GaussianBlur(edges, (5, 5), 0)

# Görsellerin karşılaştırması
cv2.imshow("Orijinal Görüntü", image)
cv2.imshow("Pürüz Algılama", blurred)

cv2.waitKey(0)
cv2.destroyAllWindows()
