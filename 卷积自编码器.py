import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np
(x_train,_),(x_test,_)=tf.keras.datasets.mnist.load_data()
x_train=np.expand_dims(x_train,-1)#再增加一个值
x_test=np.expand_dims(x_test,-1)
x_train=tf.cast(x_train,tf.float32)/255
x_test=tf.cast(x_test,tf.float32)/255
x_test_noise=x_test+0.5*np.random.normal(0,1,size=x_test.shape)
x_train_noise=x_train+0.5*np.random.normal(0,1,size=x_train.shape)
model=keras.Sequential([
    #encode:
    tf.keras.layers.Input(shape=x_train.shape[1:]),
    keras.layers.Conv2D(16,3,activation='relu',padding='same'),
    keras.layers.MaxPooling2D(padding='same'),
    keras.layers.Conv2D(32,3,activation='relu',padding='same'),
    keras.layers.MaxPooling2D(padding='same'),
    #decode:
    keras.layers.Conv2DTranspose(16,3,strides=2,activation='relu',padding='same'),
    keras.layers.Conv2DTranspose(1,3,strides=2,activation='sigmoid',padding='same')
])
model.summary()
model.compile(optimizer='adam',loss='mse')
model.fit(x_train_noise,x_train,epochs=50,batch_size=256,shuffle=True,validation_data=(x_test,x_test))
x=model.predict(x_test_noise)
x_test_noise=x_test_noise.numpy()
n=10
plt.figure(figsize=(20,4))
for i in range(1,10):
    ax=plt.subplot(2,n,i)
    plt.imshow(x_test_noise[i].reshape(28,28))
    ax=plt.subplot(2,n,n+i)
    plt.imshow(x[i].reshape(28,28))
plt.show()