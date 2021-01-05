###幸せな状態か悲しそうな状態かを畳み込みを利用して判定する###
"""
Below is code with a link to a happy or sad dataset which contains 80 images, 40 happy and 40 sad. 
Create a convolutional neural network that trains to 100% accuracy on these images, which cancels training upon hitting training accuracy of >.999

Hint -- it will work best with 3 convolutional layers.
"""
#モジュールのインポート
import tensorflow as tf
from tensorflow import keras 
import os
import zipfile
from os import path, getcwd, chdir

# DO NOT CHANGE THE LINE BELOW. If you are developing in a local
# environment, then grab happy-or-sad.zip from the Coursera Jupyter Notebook
# and place it inside a local folder and edit the path to that location

#happy-or-sad.zipのパス指定
path = f"{getcwd()}/../tmp2/happy-or-sad.zip"

#zipファイルの解凍とクローズ
zip_ref = zipfile.ZipFile(path, 'r')
zip_ref.extractall("/tmp/h-or-s")
zip_ref.close()

# GRADED FUNCTION: train_happy_sad_model
#モデル関数の作成をしていく。
def train_happy_sad_model():
    # Please write your code only where you are indicated.
    # please do not remove # model fitting inline comments.
    
#コールバックで使用するAccuracyの設定
    DESIRED_ACCURACY = 0.999

#コールバッククラスの作成
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self,epoch,logs={}):
            if logs.get("accuracy") > DESIRED_ACCURACY:
                self.model.stop_training = True
                
#コールバックインスタンスの作成             
    callbacks = myCallback()
    
#CNNモデルの構築
    # This Code Block should Define and Compile the Model. Please assume the images are 150 X 150 in your implementation.
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64,(3,3),input_shape=[150,150,3]),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512,activation='relu'),
        tf.keras.layers.Dense(1,activation="sigmoid")
    ])
#optimizerでRMSpropを利用する場合はインポート
    from tensorflow.keras.optimizers import RMSprop
#(lr=学習率)(RMSpropの学習率は公式ドキュメント上にてlr=0.001が推奨)
    model.compile(optimizer=RMSprop(lr=0.001),loss='binary_crossentropy', metrics=['accuracy'])
        

    # This code block should create an instance of an ImageDataGenerator called train_datagen 
    # And a train_generator by calling train_datagen.flow_from_directory
    
#画像生成器(ImageDataGenerator)の利用 >> 画像の前処理や水増し、オーグメンテーションが可能に。
#ImageDataGeneratorのインポート
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

#train_datagen変数 <- インスタンスを作成(rescale=標準化,オーグメントするならここに！)
    train_datagen = ImageDataGenerator(rescale=1/255)

    # Please use a target_size of 150 X 150.
#.flow_from_directoryでディレクトリから画像を加工して持ってくる。
    train_generator = train_datagen.flow_from_directory(
        "/tmp/h-or-s", #ディレクトリパスの指定
        target_size = (150,150), #画像のサイズを設定
        batch_size = 16, #バッチサイズ
        class_mode='binary') #分類するモードの設定。※複数ならcategorical
        
# Expected output: 'Found 80 images belonging to 2 classes'

    # This code block should call model.fit_generator and train for
    # a number of epochs.

# model fitting
history = model.fit_generator(
    train_generator, #generatorの指定
    steps_per_epoch=8,  #エポックごとのステップ数
    epochs=10,
    verbose=1) #verboseはトレーニング中の進捗状況や各メトリクスを表示するかどうかのパラメータ1,2でおｋ
        
    # model fitting
    return history.history['acc'][-1]
    
    
