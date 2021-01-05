"""
Exercise 3
In the videos you looked at how you would improve Fashion MNIST using Convolutions. 
For your exercise see if you can improve MNIST to 99.8% accuracy or more using only a single convolutional layer and a single MaxPooling 2D. 
You should stop training once the accuracy goes above this amount. 
It should happen in less than 20 epochs, so it's ok to hard code the number of epochs for training, 
but your training must end once it hits the above metric. If it doesn't, then you'll need to redesign your layers.

I've started the code for you -- you need to finish it!

When 99.8% accuracy has been hit, you should print out the string "Reached 99.8% accuracy so cancelling training!"
"""
#モジュールのインポート
import tensorflow as tf
from os import path, getcwd, chdir
from tensorflow import keras

# DO NOT CHANGE THE LINE BELOW. If you are developing in a local
# environment, then grab mnist.npz from the Coursera Jupyter Notebook
# and place it inside a local folder and edit the path to that location
#MNISTのパス指定
path = f"{getcwd()}/../tmp2/mnist.npz"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

#コールバッククラスの作成
class Mycallbacks(tf.keras.callbacks.Callback):
    def on_epochs_end(self,epochs,logs={}):
        if logs.get("accuracy") > 0.998:
            print("Reached 99.8% accuracy so cancelling training!")
            self.model.stop_training = True

#MNISTデータセットを畳み込みNNを利用して判定してみる
# GRADED FUNCTION: train_mnist_conv
   # YOUR CODE STARTS HERE
def train_mnist_conv():
    # Please write your code only where you are indicated.
    # please do not remove model fitting inline comments.
#MNISTのデータセットのインポート
    mnist = tf.keras.datasets.mnist
#トレーニング変数、テスト変数にデータをコピー
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data(path=path)
    # YOUR CODE STARTS HERE
#コールバックインスタンスの作成
    callbacks = Mycallbacks()
    
#画像データのreshape,グレースケール化と、標準化
    training_images=training_images.reshape(60000, 28, 28, 1)
    training_images=training_images / 255.0
    test_images = test_images.reshape(10000, 28, 28, 1)
    test_images=test_images/255.0
    # YOUR CODE ENDS HERE
#モデルの構築
    model = tf.keras.models.Sequential([
    #Conv2Dで畳み込み層の作成(n枚の畳み込み、(m*lのフィルタ),入力サイズ)
        tf.keras.layers.Conv2D(64,(3,3),activation="relu",input_shape=[28,28,1]),
    #Maxプーリング層の作成
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128,activation='relu'),
    #複数判定のため、softmaxを使用
        tf.keras.layers.Dense(10,activation="softmax")

            # YOUR CODE ENDS HERE
    ])
#モデルのコンパイル
#複数判定のため、lossに、(スパース)カテゴリカル交差エントロピー関数を適用
# YOUR CODE STARTS HERE

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # model fitting
    history = model.fit(training_images,training_labels,epochs=19,callbacks=[callbacks])

# YOUR CODE ENDS HERE

    # model fitting
    return history.epoch, history.history['acc'][-1]
