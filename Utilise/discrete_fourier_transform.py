#type:ignore
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
from scipy import fft
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import window
from scipy.ndimage.filters import gaussian_filter
import cv2
np.random.seed(42)

sig_=cv2.imread('./sdn_2.png')
sig_=cv2.resize(sig_,(300,300),cv2.INTER_AREA)
sig_=np.rollaxis(sig_, 2, 0)
image_col_one=[]
image_col_two=[]
for i in range (len(sig_)):
    sig=sig_[i,:,:]
    sig_zeros=np.zeros(sig.shape)  
    gaussian_filter(sig,1,0,sig_zeros)
    sig_one=np.zeros(sig.shape)
    gaussian_filter(sig_zeros,1,(0,1),sig_one)
    sig_two=np.zeros(sig.shape)
    gaussian_filter(sig_zeros,1,(1,0),sig_two)
    image_col_one.append(sig_one)
    image_col_two.append(sig_two)
image_ch1=np.stack([image_col_one[0],image_col_two[0]])
image_ch2=np.stack([image_col_one[1],image_col_two[1]])
image_ch3=np.stack([image_col_one[2],image_col_two[2]])
image=np.stack([image_ch1,image_ch2,image_ch3])
image=np.expand_dims(image,axis=0)

image_reshape=np.ndarray.reshape(image,(image.shape[0]*image.shape[1]*image.shape[3]*image.shape[4],image.shape[2]))
image_hann=image_reshape*window('hann',image_reshape.shape)
freqs=fft(image_hann)
sample_freq=fftpack.fftfreq(freqs.size)
power=np.abs(freqs)**2
img_power=np.ndarray.reshape(power,(image.shape[0],image.shape[2]*image.shape[1],image.shape[3],image.shape[4]))
img_freq_real=np.ndarray.reshape(np.real(freqs),(image.shape[0],image.shape[2]*image.shape[1],image.shape[3],image.shape[4]))
image_rec=img_power[0,:,:,:]
image_freq=img_freq_real[0,:,:,:]
feats=np.stack([image_rec,image_freq],axis=0)
print ('feats:',feats.shape)

