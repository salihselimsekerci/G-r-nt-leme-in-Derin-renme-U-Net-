import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_unet_model(input_shape=(128, 128, 3), num_classes=3):
    """U-Net modelini oluşturan fonksiyon.
    input_shape: Giriş görüntü boyutu (yükseklik, genişlik, kanal sayısı).
    num_classes: Segmentasyon için sınıf sayısı (arka plan dahil).
    """
    inputs = keras.Input(shape=input_shape)

    # Encoder (Aşağı Örnekleme)
    # Blok 1
    x = layers.Conv2D(64, kernel_size=3, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(64, kernel_size=3, padding="same", activation="relu")(x)
    c1 = x  # atlama bağlantısı için tutuluyor
    x = layers.MaxPooling2D(pool_size=2)(x)       # boyut yarıya (128->64)
    x = layers.Dropout(0.3)(x)
    # Blok 2
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    c2 = x
    x = layers.MaxPooling2D(2)(x)                 # boyut yarıya (64->32)
    x = layers.Dropout(0.3)(x)
    # Blok 3
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    c3 = x
    x = layers.MaxPooling2D(2)(x)                 # boyut yarıya (32->16)
    x = layers.Dropout(0.3)(x)

    # Bottleneck (En alt katman)
    x = layers.Conv2D(512, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(512, 3, padding="same", activation="relu")(x)

    # Decoder (Yukarı Örnekleme)
    # Blok 3 (Yukarı)
    x = layers.Conv2DTranspose(256, kernel_size=3, strides=2, padding="same")(x)  # boyut iki katı (16->32)
    x = layers.concatenate([x, c3])            # encoder'ın karşılık gelen çıktısıyla birleştir
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    # Blok 2 (Yukarı)
    x = layers.Conv2DTranspose(128, 3, strides=2, padding="same")(x)  # (32->64)
    x = layers.concatenate([x, c2])
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    # Blok 1 (Yukarı)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same")(x)   # (64->128)
    x = layers.concatenate([x, c1])
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)

    # Çıktı katmanı
    outputs = layers.Conv2D(num_classes, kernel_size=1, activation="softmax")(x)
    # Her piksel için num_classes boyutunda olasılık dağılımı (softmax)

    # Modeli oluştur
    model = keras.Model(inputs=inputs, outputs=outputs, name="U-Net")
    return model

# Modeli inşa et ve özetini yazdır
model = build_unet_model(input_shape=(128,128,3), num_classes=3)
model.summary()