###################転移学習について##################

# ATTENTION: Please do not alter any of the provided code in the exercise. Only add your own code where indicated
# ATTENTION: Please do not add or remove any cells in the exercise. The grader will check specific cells based on the cell position.
# ATTENTION: Please use the provided epoch values when training.

# Import all the necessary files!
#モジュールのインポート
import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from os import getcwd

#転移学習に用いる重みデータパスの取得
path_inception = f"{getcwd()}/../tmp2/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"

# Import the inception model  
#転移学習に用いるInceptionV3のインポート
from tensorflow.keras.applications.inception_v3 import InceptionV3

# Create an instance of the inception model from the local pre-trained weights
local_weights_file = path_inception
#学習済みモデル(pre_trained_model)の構築
pre_trained_model = InceptionV3(input_shape=[150,150,3],   #入力サイズの設定
                               include_top = False,        #include_top=False とした場合は全結合層 Dense が省略され、入力画像のサイズを制限するものはなくなるため、引数 input_shape で入力の形状 shape を指定できる。
                               weights = None)　　　　　　　#weights=None の場合はすべてのパラメータがランダムな重みでモデルが生成される。

pre_trained_model.load_weights(local_weights_file)  #重みデータをpre_trained_modelにロードする

# Make all the layers in the pre-trained model non-trainable
#pre_trained_modelの各層に対して、再度学習しなくてよい命令を出す
for layer in pre_trained_model.layers:
    layer.trainable = False
    
# Print the model summary
#モデルの全体像を出力
pre_trained_model.summary()

# Expected Output is extremely large, but should end with:

#batch_normalization_v1_281 (Bat (None, 3, 3, 192)    576         conv2d_281[0][0]                 
#__________________________________________________________________________________________________
#activation_273 (Activation)     (None, 3, 3, 320)    0           batch_normalization_v1_273[0][0] 
#__________________________________________________________________________________________________
#mixed9_1 (Concatenate)          (None, 3, 3, 768)    0           activation_275[0][0]             
#                                                                 activation_276[0][0]             
#__________________________________________________________________________________________________
#concatenate_5 (Concatenate)     (None, 3, 3, 768)    0           activation_279[0][0]             
#                                                                 activation_280[0][0]             
#__________________________________________________________________________________________________
#activation_281 (Activation)     (None, 3, 3, 192)    0           batch_normalization_v1_281[0][0] 
#__________________________________________________________________________________________________
#mixed10 (Concatenate)           (None, 3, 3, 2048)   0           activation_273[0][0]             
#                                                                 mixed9_1[0][0]                   
#                                                                 concatenate_5[0][0]              
#                                                                 activation_281[0][0]             
#==================================================================================================
#Total params: 21,802,784
#Trainable params: 0
#Non-trainable params: 21,802,784

#pre_trained_modelのうち、どの層を出力層とするかを設定する
last_layer = pre_trained_model.get_layer("mixed7")
print('last layer output shape: ', last_layer.output_shape)
#出力層のアウトプット形状を格納しておく
last_output = last_layer.output 
# Expected Output:
# ('last layer output shape: ', (None, 7, 7, 768))

#コールバッククラスの作成
# Define a Callback class that stops training once accuracy reaches 97.0%
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') >= 0.97):
            print("\nReached 97.0% accuracy so cancelling training!")
            self.model.stop_training = True
#RMSpropのインポート            
from tensorflow.keras.optimizers import RMSprop

#転移学習の出力層と結合するモデルの構築
# Flatten the output layer to 1 dimension
#入力層のFlatten()の後に(出力層からのアウトプット)を付け加える。xにこれを代入
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
#隠れ層の構築を行う。今まで同様にノード数、活性化関数を設定。前の層からのアウトプットである(x)を後置しておく。これらもxに代入
x = layers.Dense(1024,activation="relu")(x)# Your Code Here)(x)
# Add a dropout rate of 0.2
#ドロップアウト層の追加。同様に前層の出力(x)を後置
x = layers.Dropout(0.2)(x)# Your Code Here)(x)                  
# Add a final sigmoid layer for classification
#同様にモデルの出力層を構築。2項分類のためsigmoidを適用
x = layers.Dense(1,activation="sigmoid")(x)# Your Code Here)(x)           

#モデルにpre_trained_modelのinput情報と、出力xを適用？
model = Model(pre_trained_model.input , x) 

#モデルのコンパイル
model.compile(optimizer = RMSprop(lr=0.0001), 
              loss = "binary_crossentropy", 
              metrics = ["accuracy"])

model.summary()

# Expected output will be large. Last few lines should be:

# mixed7 (Concatenate)            (None, 7, 7, 768)    0           activation_248[0][0]             
#                                                                  activation_251[0][0]             
#                                                                  activation_256[0][0]             
#                                                                  activation_257[0][0]             
# __________________________________________________________________________________________________
# flatten_4 (Flatten)             (None, 37632)        0           mixed7[0][0]                     
# __________________________________________________________________________________________________
# dense_8 (Dense)                 (None, 1024)         38536192    flatten_4[0][0]                  
# __________________________________________________________________________________________________
# dropout_4 (Dropout)             (None, 1024)         0           dense_8[0][0]                    
# __________________________________________________________________________________________________
# dense_9 (Dense)                 (None, 1)            1025        dropout_4[0][0]                  
# ==================================================================================================
# Total params: 47,512,481
# Trainable params: 38,537,217
# Non-trainable params: 8,975,264

###################前回までと同様。zipファイルのデータを取得して、トレーニング用・テスト用ディレクトリに分割####################
# Get the Horse or Human dataset
path_horse_or_human = f"{getcwd()}/../tmp2/horse-or-human.zip"
# Get the Horse or Human Validation dataset
path_validation_horse_or_human = f"{getcwd()}/../tmp2/validation-horse-or-human.zip"
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import zipfile
import shutil

shutil.rmtree('/tmp')
local_zip = path_horse_or_human
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/training')
zip_ref.close()

local_zip = path_validation_horse_or_human
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/validation')
zip_ref.close()

# Define our example directories and files
train_dir = '/tmp/training'
validation_dir = '/tmp/validation'

train_horses_dir = os.path.join(train_dir,"horses")
train_humans_dir = os.path.join(train_dir,"humans")
validation_horses_dir = os.path.join(validation_dir,"horses")
validation_humans_dir = os.path.join(validation_dir,"humans")

train_horses_fnames = os.listdir(train_horses_dir)
train_humans_fnames = os.listdir(train_humans_dir)
validation_horses_fnames = os.listdir(validation_horses_dir)
validation_humans_fnames = os.listdir(validation_humans_dir)

print(len(train_horses_fnames))
print(len(train_humans_fnames))
print(len(validation_horses_fnames))
print(len(validation_humans_fnames))

# Expected Output:
# 500
# 527
# 128
# 128
########################################################################################################################
#ImageDataGeneratorによるオーグメンテーションの実装
# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1.0/255.0,
                                   rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   fill_mode="nearest")

# Note that the validation data should not be augmented!
#テストデータにオーグメントはしないこと！（バリエーション増やしても意味ないし）
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Flow training images in batches of 20 using train_datagen generator
#トレーニング用ディレクトリから画像を加工して取得
train_generator = train_datagen.flow_from_directory(train_dir,
                                                   target_size=(150,150),
                                                   batch_size =20,
                                                   class_mode = "binary")     

# Flow validation images in batches of 20 using test_datagen generator
#テスト用ディレクトリから画像を加工して取得
validation_generator =  test_datagen.flow_from_directory(validation_dir,
                                                   target_size=(150,150),
                                                   batch_size =20,
                                                   class_mode = "binary")

# Run this and see how many epochs it should take before the callback
# fires, and stops training at 97% accuracy

#コールバックインスタンスの作成
callbacks = myCallback()

#モデルのフィッティング
history = model.fit_generator(train_generator,
                              validation_data = validation_generator,
                              epochs = 3,
                              steps_per_epoch = 50,
                              validation_steps = 50,
                              verbose = 1,
                              callbacks=[callbacks])

######################モデル評価の可視化#########################
%matplotlib inline
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()


plt.show()
