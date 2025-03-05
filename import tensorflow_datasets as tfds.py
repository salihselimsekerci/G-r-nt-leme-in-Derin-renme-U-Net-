import tensorflow_datasets as tfds

# Oxford-IIIT Pet veri setini TFDS ile yükle (segmentasyon etiketi için versiyon 3 gerekir)
dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)
train_ds = dataset['train']
test_ds = dataset['test']

# Veri ön işleme fonksiyonları
IMAGE_SIZE = 128

def preprocess(datapoint):
    """Görüntüleri yeniden boyutlandırır ve normalleştirir, maskeleri yeniden boyutlandırır.
       Maskedeki sınıf değerlerini 0,1,2 olacak şekilde düzenler."""
    image = tf.image.resize(datapoint['image'], (IMAGE_SIZE, IMAGE_SIZE), method='nearest')
    mask = tf.image.resize(datapoint['segmentation_mask'], (IMAGE_SIZE, IMAGE_SIZE), method='nearest')
    # Normalizasyon
    image = tf.cast(image, tf.float32) / 255.0          # [0,1] aralığına çek
    mask = tf.cast(mask, tf.int32)
    # TFDS maskeleri 1,2,3 sınıf olabilir; 0-indexli yapmak için gerekirse 1 çıkarılabilir:
    mask -= 1   # maskeyi 0,1,2 değerlerine indir (arka plan 0 olsun)
    return image, mask

# Veri setine ön işlemeyi uygula ve batch'le
train_dataset = train_ds.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)
test_dataset = test_ds.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)

# Modeli derle (compile): Loss ve metrik tanımla
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Modeli eğit
EPOCHS = 20
history = model.fit(train_dataset, 
                    epochs=EPOCHS, 
                    validation_data=test_dataset)
