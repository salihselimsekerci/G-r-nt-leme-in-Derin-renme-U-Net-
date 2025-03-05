# Görüntü İşleme İçin Derin Öğrenme (U-Net)

Bu proje, bir **U-Net** modeli kullanarak görüntü segmentasyonu yapmayı hedefler. Model, her pikselin hangi sınıfa (ör. arka plan, nesne, sınır vb.) ait olduğunu tahmin eder.

## Özellikler
- U-Net mimarisi (encoder + decoder + skip connections)  
- Eğitim (training) ve tahmin (inference) örnek kodları  
- Oxford-IIIT Pet veri setiyle veya kendi verilerinizle kullanım  
- OpenCV ile kolay görselleştirme ve maske oluşturma  

## Başlarken
1. Gerekli paketleri kurun:
   ```bash
   pip install tensorflow opencv-python numpy tensorflow-datasets

Eğitimi başlatmak için:
python src/train.py

Tahmin almak için:
python src/inference.py

Klasör Yapısı
├── src/
│   ├── unet_model.py    # U-Net tanımı
│   ├── train.py         # Eğitim kodu
│   └── inference.py     # Tahmin (inference) kodu
├── test_images/
│   └── test1.jpg
└── README.md
Lisans
Bu proje MIT Lisansı ile sunulmuştur.
