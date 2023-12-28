## Genel Kütüphaneler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
from sklearn.metrics import multilabel_confusion_matrix, roc_auc_score, roc_curve, auc

## Derin Öğrenme Kütüphaneleri

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import confusion_matrix

df = pd.read_csv("C:/Users/basak/OneDrive/Masaüstü/OSTIM TECH. UNIVERSITY/4. SINIF/Yapay Zeka/data/fer2013.csv")

df.head()

print(df.shape)

df.Usage.value_counts()

df.isnull().sum()

df['emotion'].value_counts()

emotion_map = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
emotion_counts = df['emotion'].value_counts(sort=False).reset_index()
emotion_counts.columns = ['emotion', 'number']
emotion_counts['emotion'] = emotion_counts['emotion'].map(emotion_map)

# Plotting a bar graph of the class distributions
plt.figure(figsize=(6,4))
sns.barplot(x=emotion_counts.emotion, y=emotion_counts.number)
plt.title('Class distribution')
plt.ylabel('Number', fontsize=12)
plt.xlabel('Emotions', fontsize=12)
plt.show()


train_data_dir = "C:/Users/basak/OneDrive/Masaüstü/OSTIM TECH. UNIVERSITY/4. SINIF/Yapay Zeka/data/train"
validation_data_dir = "C:/Users/basak/OneDrive/Masaüstü\OSTIM TECH. UNIVERSITY/4. SINIF/Yapay Zeka/data/test"

## Resmin boyutu: 48*48 piksel
picture_size = 48

folder_path = "C:/Users/basak/OneDrive/Masaüstü/OSTIM TECH. UNIVERSITY/4. SINIF/Yapay Zeka/data/train"

expression = 'happy'

plt.figure(figsize=(12, 12))
for i in range(1, 10, 1):
    plt.subplot(3, 3, i)
    img = load_img(folder_path + '/' + expression + "/" +
                   os.listdir(folder_path + '/' + expression)[i], target_size=(picture_size, picture_size))
    plt.imshow(img)
plt.show()
## Çıktı https://puu.sh/JV9vP/edaf3d28bf.png

## Kaç farklı duygu sınıfı olduğunu tanımlıyoruz
num_classes = 7

## Görüntü boyutunu tanımlıyoruz
img_rows, img_cols = 48, 48

## Batch'i tanımlıyoruz
batch_size = 64

train_datagen = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=30,  # Görüntüleri rastgele 30 derece döndürür.
                    shear_range=0.3,    # Görüntüleri rastgele genişliklerinin veya yüksekliklerinin 0,3 katı kadar eğer
                    zoom_range=0.3,     # Görüntüleri rastgele orijinal boyutlarının 0,3 katına kadar yakınlaştırır/uzaklaştırır
                    width_shift_range=0.4,
                    height_shift_range=0.4,
                    horizontal_flip=True,
                    fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
                    train_data_dir,
                    color_mode='grayscale',
                    target_size=(img_rows,img_cols),
                    batch_size=batch_size,
                    class_mode='categorical',
                    shuffle=True)

validation_generator = validation_datagen.flow_from_directory(
                            validation_data_dir,
                            color_mode='grayscale',
                            target_size=(img_rows,img_cols),
                            batch_size=batch_size,
                            class_mode='categorical',
                            shuffle=True)


# train_datagen ve validation_datagen eğitim ve doğrulama veri kümelerini
# hazırlamak için kullanılan ImageDataGenerator nesneleridir.
# train_generator ve validation_generator eğitim ve doğrulama veri kümelerini
# içeren veri oluşturucularıdır.

model = Sequential()

# Block-1

model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(img_rows,img_cols,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(img_rows,img_cols,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Block-2

model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Block-3

model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Block-4

model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Block-1 / Block-4: Bunlar evrişim(convolutional) bloklarıdır. Her biri evrişim katmanları,
# aktivasyon fonksiyonları(ELU), parti normalleştirme, en büyük havuzlama(max-pooling) ve
# bırakma(dropout) katmanları içerir. Evrişim katmanları giriş görüntülerinde özellikleri tespit eder.

# Block-5

model.add(Flatten())
model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Blok-5, önceki evrişim katmanlarının öğrendiği özellikleri düzleştiriyor ve
# ardından tam bağlantılı katmanlarla işleme devam ediyor.

# Block-6

model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Block-7

model.add(Dense(num_classes,kernel_initializer='he_normal'))
model.add(Activation('softmax'))

# Block-6 ve Block-7: Bunlar tam bağlantılı(dense) katmanlardır ve aktivasyon fonksiyonları,
# parti normalleştirme ve bırakma katmanları içerir. Bu katmanlar, evrişim katmanları tarafından
# öğrenilen uzaysal bilgileri birleştirir.

# Çıkış katmanı çok sınıflı sınıflandırma problemleri için uygun olan softmax aktivasyon fonksiyonuna sahiptir.

print(model.summary())

from keras.optimizers import RMSprop,SGD,Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

checkpoint = ModelCheckpoint('Emotion_little_vgg.h5',
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True,
                             verbose=1)

earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=3,
                          verbose=1,
                          restore_best_weights=True
                          )

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=3,
                              verbose=1,
                              min_delta=0.0001)

callbacks = [earlystop,checkpoint,reduce_lr]
# Geribildirimler (Callbacks):
# ModelCheckpoint: Doğrulama kaybına dayanarak en iyi modeli kaydeder.
# EarlyStopping: 3 epoch boyunca doğrulama kaybında iyileşme olmazsa eğitimi durdurur ve en iyi ağırlıkları geri yükler.
# ReduceLROnPlateau: Doğrulama kaybı durgunsa öğrenme hızını azaltır. Accuracy(kesinlik) sabitse yavaşlatarak tek tek işler
## Epoch sayısı, eğitim sürecinin kaç kez tekrarlanacağını belirler.



model.compile(loss='categorical_crossentropy', #Model, kategorik çapraz entropi kaybını, 0.001 öğrenme hızına sahip Adam optimizer'ı ve doğruluk metriğini kullanarak derlenir.
              optimizer = Adam(lr=0.001),
              metrics=['accuracy'])

# Adam algoritması, eğitim boyunca öğrenme oranını ayarlar. Öğrenme oranı, model için optimal
# ağırlıkların ne kadar hızlı hesaplandığını belirler.

nb_train_samples = 24320
nb_validation_samples = 3072

#nb_train_samples ve nb_validation_samples, eğitim ve doğrulama örnek sayısını temsil eder.

epochs=40

#epochs 40 olarak ayarlanmıştır.

history=model.fit_generator(
                train_generator,
                steps_per_epoch=nb_train_samples//batch_size,
                epochs=epochs,
                callbacks=callbacks,
                validation_data=validation_generator,
                validation_steps=nb_validation_samples//batch_size)

model.load_weights('Emotion_little_vgg.h5')

score = model.evaluate_generator(validation_generator)

print('Doğruluk:', score[1])

y_pred = model.predict_generator(validation_generator)

y_pred = np.argmax(y_pred, axis=1)

y_true = validation_generator.classes

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(12, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.show()

plt.style.use('dark_background')

plt.figure(figsize=(20,10))
plt.subplot(1, 2, 1)
plt.suptitle('Optimizer : Adam', fontsize=10)
plt.ylabel('Loss', fontsize=16)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')

plt.subplot(1, 2, 2)
plt.ylabel('Accuracy', fontsize=16)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.show()

# ROC eğrisini çizmek için sınıf etiketleri ve tahmin olasılıklarını al
y_true = np.array(validation_generator.classes)
y_score = model.predict_generator(validation_generator)

# ROC eğrisini çizimi
plt.figure(figsize=(12, 12))
for i in range(num_classes):
    fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')

plt.title('ROC Curves for Each Class')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()
