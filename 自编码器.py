import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
(x_train,_),(x_test,_)=tf.keras.datasets.mnist.load_data()
x_train=x_train.reshape(x_train.shape[0],-1)
x_test=x_test.reshape(x_test.shape[0],-1)#将矩阵转为向量
x_train=tf.cast(x_train,tf.float32)/255
x_test=tf.cast(x_test,tf.float32)/255
inputn=tf.keras.layers.Input(shape=(784,))
encode=tf.keras.layers.Dense(32,activation='relu')(inputn)
decode=tf.keras.layers.Dense(784,activation='sigmoid')(encode)
model=tf.keras.Model(inputn,decode)
model.summary()
model.compile(optimizer='adam',loss='mse')
model.fit(x_train,x_train,epochs=50,batch_size=256,shuffle=True,validation_data=(x_test,x_test))
image=model.predict(x_test)
x_test=x_test.numpy()
n=10
plt.figure(figsize=(20,4))
for i in range(1,10):
    ax=plt.subplot(2,n,i)
    plt.imshow(x_test[i].reshape(28,28))
    ax=plt.subplot(2,n,n+i)
    plt.imshow(image[i].reshape(28,28))
plt.show()
print(x_test.shape)