"""
Implementation of Denoising AutoEncoder
"""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as pimg
import seaborn as sns
import os

import tensorflow as tf
from tensorflow.keras.datasets import cifar10,mnist
from tensorflow.keras.optimizers import Adam
import argparse
import cv2
from tensorflow.keras.layers import BatchNormalization,Conv2D,Conv2DTranspose,LeakyReLU,Activation,Flatten,Dense,Reshape,Input
from tensorflow.keras.models import Model,load_model

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# Creating Model
class DAE():
	@staticmethod
	def buildmodel(height,width,depth,filters=(32,64),latentDim=32):
		
		# Encoder Architecture
		inputs = Input(shape=(height,width,depth))
		x = inputs
		for f in filters:
			x = Conv2D(f, (3,3), strides=2, padding='same')(x)
			x = LeakyReLU(alpha=0.2)(x)
			x = BatchNormalization(axis=-1)(x)
			
		volumeSize = tf.keras.backend.int_shape(x)
		x = Flatten()(x)
		latent = Dense(latentDim)(x)
		
		# Encoder Model
		Encoder = Model(inputs,latent,name="Encoder")
		
		
		# Decoder Architecture
		latentInputs = Input(shape=(latentDim,))
		x = Dense(np.prod(volumeSize[1:]))(latentInputs)
		x = Reshape((volumeSize[1],volumeSize[2],volumeSize[3]))(x)
		for f in filters[::-1]:
			x = Conv2DTranspose(f, (3,3), strides=2, padding='same')(x)
			x = LeakyReLU(alpha=0.2)(x)
			x = BatchNormalization(axis=-1)(x)
		x = Conv2DTranspose(depth, (3,3), padding='same')(x)
		output = Activation("sigmoid")(x)
		
		# Decoder Model
		Decoder = Model(latentInputs,output,name="Decoder")
		
		# Encoder and Decoder --> AutoEncoder
		AutoEncoder = Model(inputs,Decoder(Encoder(inputs)),name="AutoEncoder")
		
		return (Encoder,Decoder,AutoEncoder)
		
dae = DAE()
Encoder,Decoder,AutoEncoder = dae.buildmodel(28,28,1)
print (AutoEncoder.summary())
tf.keras.utils.plot_model(Encoder, to_file='Images/Encoder_MNIST.png', show_shapes=True, show_layer_names=True)
tf.keras.utils.plot_model(Decoder, to_file='Images/Decoder_MNIST.png', show_shapes=True, show_layer_names=True)
tf.keras.utils.plot_model(AutoEncoder, to_file='Images/AutoEncoder_MNIST.png', show_shapes=True, show_layer_names=True)

# Data Preparation
((Y_Train,_),(Y_Test,_)) = mnist.load_data()

# Changing Channels of Image from 2(i.e Gray Scale) to 3
Y_Train = np.expand_dims(Y_Train,axis=-1)
Y_Test = np.expand_dims(Y_Test,axis=-1)

Y_Train = Y_Train.astype("float32")/255.0
Y_Test = Y_Test.astype("float32")/255.0

X_Train_Noise = np.random.normal(loc=0.5, scale=0.5, size=Y_Train.shape)
X_Test_Noise = np.random.normal(loc=0.5, scale=0.5, size=Y_Test.shape)

X_Train = np.clip(Y_Train + X_Train_Noise,0,1)
X_Test = np.clip(Y_Test + X_Test_Noise,0,1)

# Plotting Data
for i in range(10):
	plt.subplot(2, 5, 1 + i)
	plt.axis('off')
	if i<5:
		plt.imshow((X_Train[i][:,:,0]),cmap='gray')
	else:
		plt.imshow((Y_Train[i-5][:,:,0]),cmap='gray')
plt.show()


# Training
AutoEncoder.compile(loss='mse', optimizer='adam')
Epochs=100
Batch_Size=32

'''
AutoEncoder = load_model('Models/DAE_Model_MNIST.h5')
'''
Train_History = AutoEncoder.fit(X_Train,Y_Train,validation_data=(X_Test, Y_Test),epochs=Epochs,batch_size=Batch_Size)
AutoEncoder.save("Models/DAE_Model_MNIST.h5")

# Plotting Train Results
N = np.arange(Epochs)
plt.style.use("seaborn-poster")
plt.figure()
plt.plot(N, Train_History.history["loss"], label="Training Loss")
plt.plot(N, Train_History.history["val_loss"], label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(loc="best")
plt.savefig("Images/Loss_MNIST.png")


# Testing
AutoEncoder = load_model('Models/DAE_Model_MNIST.h5')
for i in range(15):
	plt.subplot(3, 5, 1 + i)
	plt.axis('off')
	if i<5:
		plt.imshow((X_Train[i][:,:,0]),cmap='gray')
	elif i<10 and i>=5:
		plt.imshow((Y_Train[i-5][:,:,0]),cmap='gray')
	else:
		X = np.expand_dims(X_Train[i-10],axis=0)
		Y = AutoEncoder.predict(X)
		Y = Y[0,:,:,0]
		plt.imshow(Y,cmap='gray')
		
plt.savefig("Images/Results_MNIST.png")
plt.show()
