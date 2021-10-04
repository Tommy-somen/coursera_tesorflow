"""
In this exercise you'll try to build a neural network that predicts the price of a house according to a simple formula.

So, imagine if house pricing was as easy as a house costs 50k + 50k per bedroom, so that a 1 bedroom house costs 100k, a 2 bedroom house costs 150k etc.

How would you create a neural network that learns this relationship so that it would predict a 7 bedroom house as costing close to 400k etc.

Hint: Your network might work better if you scale the house price down. You don't have to give the answer 400...it might be better to create something that predicts the number 4, and then your answer is in the 'hundreds of thousands' etc.
"""
                                                                 
#モジュールのインポート
import tensorflow as tf
import numpy as np
from tensorflow import keras

# GRADED FUNCTION: house_model
def house_model(y_new):
    xs = np.array([1.0,2.0,3.0,4.0,5.0,6.0])
    ys = np.array([1.0,1.5,2.0,2.5,3.0,3.5])
    #モデルの構築。Dense(units数,入力する形状=[shape])
    model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    #モデルのコンパイル(最適化関数(optimizer)="~~", 損失関数(loss)="~~")
    model.compile(optimizer="sgd",loss="mean_squared_error")
    #データへのフィッティング(トレーニングデータ、正解データ、エポック数(epochs)="num")
    model.fit(xs,ys,epochs=500)
    return model.predict(y_new)[0]

prediction = house_model([7.0])
print(prediction)

"""
Expected
Epoch 1/500
6/6 [==============================] - 2s 363ms/sample - loss: 14.7348
Epoch 2/500
6/6 [==============================] - 0s 284us/sample - loss: 6.8722
...
Epoch 499/500
6/6 [==============================] - 0s 222us/sample - loss: 0.0026
Epoch 500/500
6/6 [==============================] - 0s 217us/sample - loss: 0.0026
[4.07325]
"""




