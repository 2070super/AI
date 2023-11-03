import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
batch_size = 64
def generator():
    model=tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(256,input_shape=(100,),use_bias=False))
    model.add(tf.keras.BatchNormalization())
    model.add(tf.keras.LeakyReLU())
    model.add(tf.keras.layers.Dense(512,  use_bias=False))
    model.add(tf.keras.BatchNormalization())
    model.add(tf.keras.LeakyReLU())
    model.add(tf.keras.layers.Dense(28*28*1,use_bias=False,activation='tanh'))
    model.add(tf.keras.BatchNormalization())
    model.add(tf.keras.layers.Reshape(28,28,1))
    return model
def discriminator():
    model=tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(512,use_bias=False))
    model.add(tf.keras.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Dense(256,use_bias=False))
    model.add(tf.keras.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Dense(1))
    return model
def discriminator_loss(real_out,fake_out):
    cross_entropy = tf.keras.losses.binary_crossentropy(from_logits=True)
    image_real_loss=cross_entropy(0.9*tf.ones_like(real_out),real_out)
    image_fake_loss=cross_entropy(tf.zeros_like(fake_out),fake_out)
    return image_fake_loss+image_real_loss
def generator_loss(fake_out):
    cross_entropy = tf.keras.losses.binary_crossentropy(from_logits=True)
    return cross_entropy(0.9*tf.ones_like(fake_out),fake_out)
generator_optimizer=tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer=tf.keras.optimizers.Adam(1e-4)
def init():
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    print(train_images.shape)
    train_images = tf.expand_dims(train_images, -1)
    train_images = tf.cast(train_images, tf.float32)
    train_images = (train_images - 127.5) / 127.5
    print(train_images[0])
    datasets = tf.data.Dataset.from_tensor_slices(train_images)
    datasets = datasets.shuffle(600000).batch(batch_size)

if __name__=='__main__':
    init()
    epoch=100
    noise_dim=100
    nsamples=20
    z=tf.random.normal([nsamples,noise_dim])
    generator=generator()
    discriminator=discriminator()

