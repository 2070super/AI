import tensorflow as tf
import pylab
import  numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
class mycallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if(logs.get('loss')<0.001):
            print("\n loss is low,so cancelling training")
            self.model.stop_training=True
clouth=keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels)=clouth.load_data()
#plt.imshow(train_images[2])
#pylab.show()
callbacks=mycallback()
train_images=train_images/255.0
test_images=test_images/255.0
model=keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation=tf.nn.relu),
    keras.layers.Dense(10,activation=tf.nn.softmax)
])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy')
model.fit(train_images,train_labels,epochs=500,callbacks=[callbacks])
model.evaluate(test_images,test_labels)