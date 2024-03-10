import io
import numpy as np
import glob
import random
import os
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from keras.models import *
from keras.layers import  Input,Conv2D,BatchNormalization,Activation,Lambda,Subtract
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from skimage.measure import compare_psnr, compare_ssim

def DnCNN():
    
    inpt = Input(shape=(None,None,1))
    # 1st layer, Conv+relu
    x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(inpt)
    x = Activation('relu')(x)
    # 15 layers, Conv+BN+relu
    for i in range(15):
        x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(x)
#         x = BatchNormalization(axis=-1, epsilon=1e-3)(x)
        x = Activation('relu')(x)   
    # last layer, Conv
    x = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding='same')(x)
    x = Subtract()([inpt, x])   # input - artifacts
    model = Model(inputs=inpt, outputs=x)
    
    return model


def step_decay(epoch,lr):
    """
    Learning rate scheduler, decrease learning rate by 10 after 50 eopchs
    
    @params:
    epoch : Num of epochs 
    lr : Learning rate
    
    Returns:
    lr : Learning rate (callback)
    """
    
    initial_lr = lr
    if epoch<50:
        lr = initial_lr
    else:
        lr = initial_lr/10

    return lr    


def train_datagen(y_, c_data, batch_size=8):
    """
    Generator to yield data to the network
    
    @params:
    y_ : tensor of clean patches
    c_data  tensor of compressed patches
    
    Returns:
    ge_batch_x, artifacts : i/p, o/p for the network
    """

    indices = list(range(y_.shape[0]))
    while(True):
        np.random.shuffle(indices)    # shuffle
        
        for i in range(0, int(len(indices)), batch_size):
            ge_batch_y = y_[indices[i:i+batch_size]] #clean images batch
            artifacts = c_data[indices[i:i+batch_size]] - y_[indices[i:i+batch_size]] #subtract the grayscale images to get the artifacts batch

            ge_batch_x = ge_batch_y + artifacts  # input image = clean image + artifacts
            yield ge_batch_x, artifacts

def process():
	train_data_clean = "train/clean_patches.npy"
	data = np.load(train_data_clean)
	print('Size of train data: ({}, {}, {})'.format(data.shape[0],data.shape[1],data.shape[2]))

	data = data.reshape((data.shape[0],data.shape[1],data.shape[2],1))
	data = data.astype('float32')/255.0

	train_data_compressed = "train/compressed_patches.npy"
	c_data = np.load(train_data_compressed)
	print('Size of train data: ({}, {}, {})'.format(data.shape[0],data.shape[1],data.shape[2]))

	print(c_data.shape)
	c_data = c_data.reshape((c_data.shape[0],c_data.shape[1],c_data.shape[2],1))
	c_data = c_data.astype('float32')/255.0


	model = DnCNN()
	model.summary()

	batch_size = 128
	lr = 0.001
	save_every = 10 #save model h5 after every "n" epochs
	epoch = 10
	
	# compile the model
	model.compile(optimizer=Adam(), loss=['mse'])
	
	# compile the model
	model.compile(optimizer=Adam(), loss=['mse'])
		
	# use call back functions
	ckpt = ModelCheckpoint('model.h5', monitor='val_loss',verbose=0, period=save_every)
		
	lr = LearningRateScheduler(step_decay)
	# train 
	history = model.fit_generator(train_datagen(data, c_data, batch_size=batch_size),steps_per_epoch=10, epochs=epoch, verbose=1,callbacks=[ckpt, lr]) #decreased the num of steps to reduce training time by dividng by 10
	                
	plt.plot(history.history['loss'])
	# plt.plot(history.history['val_loss'])
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.savefig("results/loss.png")

	plt.pause(5)
	plt.show(block=False)
	plt.close()
