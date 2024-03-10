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


def compress_image(img, quality = 10, display_image = False):
    buffer = io.BytesIO() #using a buffer in case we do not wish to save it in memory
    img.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    compressed_image = Image.open(io.BytesIO(buffer.read()))
    return compressed_image
   
    
def process(file):
	# Load model
	model = load_model("model.h5")

	clean_img = Image.open(file).convert("L")
	fig = plt.figure()
	plt.gca().set_title('Clean Image')
	plt.gca().axis('off')
	plt.imshow(clean_img,cmap=plt.cm.gray)
	plt.savefig("results/clean_img.png")

	image_file = Image.open("results/clean_img.png")
	
	s1=str(len(image_file.fp.read()))
	print("File Size In Bytes:- "+s1)
	

	plt.pause(5)
	plt.show(block=False)
	plt.close()
	
	
	
	# compress the image
	compressed_img = compress_image(clean_img, quality = 10, display_image = True) #compressed image
	fig = plt.figure()
	plt.gca().set_title('Compressed Image')
	plt.gca().axis('off')
	plt.imshow(compressed_img,cmap=plt.cm.gray)
	plt.savefig("results/compressed_img.png")
	
	image_file = Image.open("results/compressed_img.png")
	s2=str(len(image_file.fp.read()))
	print("File Size In Bytes:- "+s2)
	
	plt.pause(5)
	plt.show(block=False)
	plt.close()	
	
	# convert PIL objects (clean image and compressed image) to numpy arrays and pre-process them
	clean_img = np.array(clean_img, dtype='float32') / 255.0
	
	compressed_img = np.array(compressed_img, dtype='float32') / 255.0
	# test image will be the compressed image
	img_test = compressed_img 
	
	# predict the residual image
	img_test = img_test.reshape(1, img_test.shape[0], img_test.shape[1], 1) 
	residual_image = model.predict(img_test)
	residual_image = residual_image.reshape((compressed_img.shape[0], compressed_img.shape[1]))
	
	# subtract the residual image from the compressed image
	restored_image = compressed_img - residual_image
	fig = plt.figure()
	plt.gca().set_title('Restored Image')
	plt.gca().axis('off')
	plt.imshow(restored_image, cmap=plt.cm.gray)
	plt.savefig("results/restored_image.png")

	image_file = Image.open("results/restored_image.png")
	s3=str(len(image_file.fp.read()))
	print("File Size In Bytes:- "+s3)

	plt.pause(5)
	plt.show(block=False)
	plt.close()
	print(s1)
	print(s2)
	print(s3)
	return s1,s2,s3
	