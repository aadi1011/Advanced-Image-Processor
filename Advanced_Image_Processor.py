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
