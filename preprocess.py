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


patch_size, stride = 50, 10 
save_dir = 'train/'
aug_times = 1
patches = []


def data_aug(img, mode=0):
    """Data augmentation
    1. The flipud() function is used to flip an given array in the up/down direction.
    2. np.rot90 -> k : Number of times the array is rotated by 90 degrees.
    """
    
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))


def compress_image(img, quality = 10, display_image = False):
    """ Perform JPEG compression
    @params:
    img : PIL object
    quality :  Quality values must be in the range [0, 100]. Small quality values result in more compression and stronger compression artifacts.
    
    Returns:
    PIL compressed image object
    """
    buffer = io.BytesIO() #using a buffer in case we do not wish to save it in memory
    img.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    compressed_image = Image.open(io.BytesIO(buffer.read()))
    return compressed_image


def process(path):
	file_list = glob.glob(path+"/*.png") # Train data
	print("{} images found in train folder".format(len(file_list)))
	for i in range(0,len(file_list)):
		file_name=file_list[i]
		print(file_name)
		img = cv2.imread(file_name, 0)  # gray scale
		img = cv2.resize(img, (250,250), interpolation=cv2.INTER_CUBIC)
		h, w = img.shape
		scales = [1, 0.9, 0.8, 0.7]
		for s in scales:
			h_scaled, w_scaled = int(h*s),int(w*s)
			img_scaled = cv2.resize(img, (h_scaled,w_scaled), interpolation=cv2.INTER_CUBIC)
			# extract patches
			for i in range(0, h_scaled-patch_size+1, stride):
				for j in range(0, w_scaled-patch_size+1, stride):
					x = img_scaled[i:i+patch_size, j:j+patch_size]
					# data aug
					for k in range(0, aug_times):
						x_aug = data_aug(x, mode=np.random.randint(0,8))
						patches.append(x_aug)
	# save to .npy
	res = np.array(patches)
	print('Shape of result = ' + str(res.shape))
	print('Saving data...')
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	np.save(save_dir+'clean_patches.npy', res)
	train_data_clean = "train/clean_patches.npy"
	data = np.load(train_data_clean)

	c_data = [] #np.zeros(data.shape)
	for idx, img in enumerate(data[:]):
		print(idx)
		img = Image.fromarray(img)
		q = random.choice([5,10,50,90,100])
		c_img = compress_image(img, quality = q, display_image = False) #compressed image
		c_img = np.array(c_img)
		c_data.append(c_img)

	c_data = np.array(c_data)
	print('Shape of result = ' + str(c_data.shape))
	print('Saving data...')
	np.save('train/compressed_patches.npy', c_data)
