"""
Exercise 2
In the course you learned how to do classificaiton using Fashion MNIST, a data set containing items of clothing. There's another, similar dataset called MNIST which has items of handwriting -- the digits 0 through 9.

Write an MNIST classifier that trains to 99% accuracy or above, and does it without a fixed number of epochs -- i.e. you should stop training once you reach that level of accuracy.

Some notes:

It should succeed in less than 10 epochs, so it is okay to change epochs= to 10, but nothing larger
When it reaches 99% or greater it should print out the string "Reached 99% accuracy so cancelling training!"
If you add any additional variables, make sure you use the same names as the ones used in the class
I've started the code for you below -- how would you finish it?
"""

#モジュールのインポート
import tensorflow as tf
from tensorflow import keras
from os import path, getcwd, chdir
import matplotlib.pyplot as plt

# DO NOT CHANGE THE LINE BELOW. If you are developing in a local
# environment, then grab mnist.npz from the Coursera Jupyter Notebook
# and place it inside a local folder and edit the path to that location
#MNISTデータセットの場所を指定
path = f"{getcwd()}/../tmp2/mnist.npz"

#コールバックを行うクラスの作成
class Callbacks(tf.keras.callbacks.Callback):
    def epochs_end(self,epoch,log={}):
        if log.get("acc")>=0.990:
            self.model.stop_training = True
            
# GRADED FUNCTION: train_mnist
def train_mnist():
    # Please write your code only where you are indicated.
    # please do not remove # model fitting inline comments.

    # YOUR CODE SHOULD START HERE
    #コールバックインスタンスの作成
    callbacks = Callbacks()

    # YOUR CODE SHOULD END HERE

    mnist = tf.keras.datasets.mnist

    (x_train, y_train),(x_test, y_test) = mnist.load_data(path=path)
    # YOUR CODE SHOULD START HERE
    #トレーニングデータの標準化
    x_train = x_train/ 255.0
    #トレーニングデータの標準化
    x_test = x_test/ 255.0
    # YOUR CODE SHOULD END HERE
    #モデルの構築
    # YOUR CODE SHOULD START HERE
    model = tf.keras.models.Sequential([keras.layers.Flatten(input_shape=[28,28]),
                                        keras.layers.Dense(512,activation=tf.nn.relu),
                                        keras.layers.Dense(512,activation=tf.nn.relu),
                                        keras.layers.Dense(10,activation=tf.nn.softmax)]
    
    # YOUR CODE SHOULD END HERE

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # model fitting
    history = model.fit(x_train,y_train,epochs=9,callbacks=[callbacks])# YOUR CODE SHOULD START HERE
              # YOUR COdE SHOULD END HERE
    # model fitting
    return history.epoch, history.history['acc'][-1]
