# Advanced Image Processing Application that performs various techniques on a given input image. 
# Developer: Aadith Sukumar (https://github.com/aadi1011)

#################### IMPORTS ####################

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
from streamlit_extras.switch_page_button import switch_page 

#################### ALL FUNCTIONS ####################

def plot_image(img):
    fig = plt.figure(figsize=(10,10))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    return fig

def plot_histogram(img):
    fig = plt.figure(figsize=(10,10))
    plt.hist(img.ravel(), 256, [0,256])
    plt.title("Histogram")
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")
    return fig

def FDF_lowpass(img, D0, n):
    # Converting the image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Performing the Fourier Transform
    M, N = img.shape # image size
    H = np.zeros((M, N), dtype=np.float32) # filter

    # create the filter
    for u in range(0, M):
        for v in range(0, N):
            D = np.sqrt((u - M/2)**2 + (v - N/2)**2)
            H[u, v] = 1 / (1 + (D/D0)**(2*n))

    # apply the filter
    img_fft = np.fft.fft2(img)
    img_fft_shift = np.fft.fftshift(img_fft)
    img_fft_shift_filt = img_fft_shift * H
    img_fft_filt = np.fft.ifftshift(img_fft_shift_filt)
    img_filt = np.fft.ifft2(img_fft_filt)
    img_filt = np.abs(img_filt)

    # Converting the image back to RGB
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_filt = cv2.cvtColor(img_filt.astype('float32'), cv2.COLOR_GRAY2RGB)

    return img_filt

def FDF_highpass(img, D0, n):
    # Converting the image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Performing the Fourier Transform
    M, N = img.shape # image size
    H = np.zeros((M, N), dtype=np.float32) # filter
    
    # create the filter
    for u in range(0, M):
        for v in range(0, N):
            D = np.sqrt((u - M/2)**2 + (v - N/2)**2)
            H[u, v] = 1 / (1 + (D0/D)**(2*n))

    # apply the filter
    img_fft = np.fft.fft2(img)
    img_fft_shift = np.fft.fftshift(img_fft)
    img_fft_shift_filt = img_fft_shift * H
    img_fft_filt = np.fft.ifftshift(img_fft_shift_filt)
    img_filt = np.fft.ifft2(img_fft_filt)
    img_filt = np.abs(img_filt)

    # Converting the image back to RGB
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_filt = cv2.cvtColor(img_filt.astype('float32'), cv2.COLOR_GRAY2RGB)

    return img_filt

def FDF_blur(img, kernel_size):
    blur_img = cv2.blur(img, (kernel_size, kernel_size))
    return blur_img

def FDF_sharpen(img, kernel_size):
    kernel_size = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpen_img = cv2.filter2D(img, -1, kernel_size)
    return sharpen_img

def FDF_gaussian_noise(img, mean, stddev):
    #mean = img.mean()
    #stddev = img.std()
    noise = np.random.normal(mean, stddev, img.shape)
    noise_img = img + noise
    return noise_img

def FDF_salt_pepper_noise(img, val):
    sp_noise = np.random.rand(*img.shape) < val
    img_SP = img.copy()
    img_SP[sp_noise] = 255
    return img_SP

def ED_sobel(img, kernel_size):
    sobelx = cv2.Sobel(np.float32(img), cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobely = cv2.Sobel(np.float32(img), cv2.CV_64F, 0, 1, ksize=kernel_size)
    sobelxy = np.sqrt(sobelx**2 + sobely**2)
    return sobelxy

def ED_prewitt(img):
    prewittx = cv2.filter2D(img, -1, np.array([[-1,0,1], [-1,0,1], [-1,0,1]]))
    prewitty = cv2.filter2D(img, -1, np.array([[-1,-1,-1], [0,0,0], [1,1,1]]))
    prewittxy = np.sqrt(prewittx**2 + prewitty**2)
    return prewittxy

def ED_roberts(img):
    robertsx = cv2.filter2D(img, -1, np.array([[0, 0, 0], [0, 1, 0], [0, 0, -1]]))
    robertsy = cv2.filter2D(img, -1, np.array([[0, 0, 0], [0, 0, 1], [0, -1, 0]]))
    robertsxy = np.sqrt(robertsx**2 + robertsy**2)
    return robertsxy

def ED_canny(img, upper_threshold, lower_threshold):
    canny_img = cv2.Canny(img, upper_threshold, lower_threshold)
    return canny_img
