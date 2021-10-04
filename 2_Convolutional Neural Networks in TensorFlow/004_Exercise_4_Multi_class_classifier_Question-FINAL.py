########################マルチクラス分類_CNN#########################


# ATTENTION: Please do not alter any of the provided code in the exercise. Only add your own code where indicated
# ATTENTION: Please do not add or remove any cells in the exercise. The grader will check specific cells based on the cell position.
# ATTENTION: Please use the provided epoch values when training.
#モジュールのインポート
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from os import getcwd

#csvファイルからのデータセット取得関数
def get_data(filename):
    with open(filename) as training_file:
        reader = csv.reader(training_file,delimiter=',')
        imgs = []
        labels = []
        
        #1行目はカラムのタイトルなので、1行スキップさせる
        next(reader,None)
        
        #行ごとにラベルと、画像データを取得していく。
        for row in reader:
            label = row[0]
            data = row[1:]
            #画像データはそのままではCNNに突っ込めないので、np.array()でnumpyに変換→好きな画像サイズにreshape
            img = np.array(data).reshape((28,28))
            
            imgs.append(img)
            labels.append(label)
        #ラベルと画像データをnumpy変換して、float型にする
        images = np.array(imgs).astype(float)
        labels = np.array(labels).astype(float)
        return images, labels
        
print(training_images.shape)
print(training_labels.shape)
print(testing_images.shape)
print(testing_labels.shape)
# Their output should be:
# (27455, 28, 28)
# (27455,)
# (7172, 28, 28)
# (7172,)
        
        
# In this section you will have to add another dimension to the data
# So, for example, if your array is (10000, 28, 28)
# You will need to make it (10000, 28, 28, 1)
# Hint: np.expand_dims

#画像データに1次元追加するnp.expand_dimsを使用
training_images =  np.expand_dims(training_images,axis=3)
testing_images = np.expand_dims(testing_images,axis=3)

#ImageDataGeneratorで標準化、オーグメントを実施
# Create an ImageDataGenerator and do Image Augmentation
train_datagen = ImageDataGenerator(rescale=1.0/255.0,
                                   rotation_range=40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   fill_mode = "nearest"
    )

#テストデータも同様に。
validation_datagen = ImageDataGenerator(rescale=1.0/255.0)
    
# Keep These
print(training_images.shape)
print(testing_images.shape)   
# Their output should be:
# (27455, 28, 28, 1)
# (7172, 28, 28, 1)

#CNNモデルの構築
# Define the model
# Use no more than 2 Conv2D and 2 MaxPooling2D
model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(32,(3,3),activation="relu",input_shape=(28,28,1)),#reshapeしたサイズに合わせること
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Conv2D(32,(3,3),activation="relu"),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(1024,activation="relu"),
                                    tf.keras.layers.Dense(26,activation="softmax")]) #複数分類はsoftmaxで

#モデルのコンパイル
# Compile Model. 
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy", #複数分類なのでカテゴリカル系を使用
              metrics = ["accuracy"])

train_generator = train_datagen.flow(training_images, 
                                     training_labels,
                                     batch_size = 64)

validation_generator = validation_datagen.flow(testing_images,
                                              testing_labels,
                                              batch_size = 64)

# Train the Model
#モデルのフィッティング
history = model.fit_generator(train_generator,
                              epochs = 20,
                              validation_data = validation_generator)#validation_steps=50)

model.evaluate(testing_images, testing_labels, verbose=0)


#####################モデル評価の可視化#########################
# Plot the chart for accuracy and loss on both training and validation
%matplotlib inline
import matplotlib.pyplot as plt
acc = history.history['accuracy']# Your Code Here
val_acc = history.history['val_accuracy']# Your Code Here
loss = history.history['loss']# Your Code Here
val_loss = history.history['val_loss']# Your Code Here

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
