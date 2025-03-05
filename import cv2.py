import cv2
import numpy as np

# Test etmek istediğiniz görüntüyü okuyun (örnek olarak 'test1.jpg')
image_bgr = cv2.imread('test1.jpg')  # OpenCV görüntüyü BGR formatında okur
if image_bgr is None:
    raise FileNotFoundError("Görüntü bulunamadı, lütfen doğru path sağlayın.")

# Modelin girdi boyutuna yeniden boyutlandır
input_img = cv2.resize(image_bgr, (IMAGE_SIZE, IMAGE_SIZE))
# Gerekirse renk formatını dönüştür (BGR -> RGB, model eğitimi RGB kabul ettiyse)
input_img_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
# [0,1] aralığına normalize et
input_tensor = input_img_rgb.astype(np.float32) / 255.0
# Batch boyutunu ekle ve modele ver
input_tensor = np.expand_dims(input_tensor, axis=0)  # shape = (1, 128, 128, 3)
pred = model.predict(input_tensor)  # shape = (1, 128, 128, 3)
# Her piksel için en yüksek olasılık sınıfını al (argmax)
pred_mask = np.argmax(pred, axis=-1)  # shape = (1, 128, 128)
pred_mask = pred_mask[0]             # shape = (128, 128)

# Segmentasyon maskesini görselleştir: sınıf indekslerini renklere çevir
# Renk paleti (BGR formatında): 0->yeşil, 1->mor, 2->sarı
color_map = {
    0: (0, 255, 0),    # Arka plan - Yesil
    1: (128, 0, 128),  # Hayvan - Mor (128,0,128 BGR formatında mordur)
    2: (0, 255, 255)   # Sınır - Sarı (0,255,255 BGR -> mavi 0, yesil 255, kirmizi 255)
}
mask_color = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
for cls, color in color_map.items():
    mask_color[pred_mask == cls] = color

# Orijinal görüntüyle aynı boyuta büyüt (gerekirse)
mask_color_full = cv2.resize(mask_color, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

# Sonucu dosyaya yaz (renkli maske olarak)
cv2.imwrite('test1_mask_output.png', mask_color_full)
