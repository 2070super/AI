import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
def generator():
    model=tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(256,input_shape=(100,),use_bias=False))
    model.add(tf.keras.BatchNormalization())
    model.add(tf.keras.LeakyReLU())
    model.add(tf.keras.layers.Dense(512,  use_bias=False))
    model.add(tf.keras.BatchNormalization())
    model.add(tf.keras.LeakyReLU())
    model.add(tf.keras.layers.Dense(28*28*1,use_bias=False,activation='tanh')
    model.add(tf.keras.BatchNormalization())
    model.add(tf.keras.layers.Reshape(28,28,1))
    return model
def discriminator():
    model=tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())
    
def init():
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    print(train_images.shape)
    train_images = tf.expand_dims(train_images, -1)
    train_images = tf.cast(train_images, tf.float32)
    train_images = (train_images - 127.5) / 127.5
    print(train_images[0])
    batch_size = 64
    datasets = tf.data.Dataset.from_tensor_slices(train_images)
    datasets = datasets.shuffle(600000).batch(batch_size)


