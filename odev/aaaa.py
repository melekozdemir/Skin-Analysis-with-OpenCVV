from tensorflow.keras.models import load_model

# compile=False ile modeli yükle
model = load_model(r'C:\Users\mlkoz\OneDrive\Desktop\Python\odev\modell.h5', compile=False)

# Modelin özetini yazdır
model.summary()
