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

def PSNR(img1, img2):
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
    return psnr

def IS_binary_thresholding(img, threshold):
    ret, thresh_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return thresh_img

def IS_inverse_binary_thresholding(img, threshold):
    ret, thresh_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY_INV)
    return thresh_img

def IS_truncated_thresholding(img, threshold):
    ret, thresh_img = cv2.threshold(img, threshold, 255, cv2.THRESH_TRUNC)
    return thresh_img

def IS_to_zero_thresholding(img, threshold):
    ret, thresh_img = cv2.threshold(img, threshold, 255, cv2.THRESH_TOZERO)
    return thresh_img

def IS_otsu_thresholding(img):
    ret, thresh_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh_img

def IS_gaussian_thresholding(img):
    ret, thresh_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh_img

def IS_mean_adaptive_thresholding(img, block_size, constant):
    thresh_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, constant)
    return thresh_img

def IS_gaussian_adaptive_thresholding(img, block_size, constant):
    thresh_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, constant)
    return thresh_img

def RBM_region_growing(img, seed, threshold):
    neighbours = [(0,1), (1,0), (0,-1), (-1,0)]
    region_threshold = threshold
    height, width = img.shape
    visited = np.zeros_like(img)
    segmented_img = np.zeros_like(img)
    stack = []
    stack.append(seed)
    seed_intensity = img[seed]
    while len(stack) > 0:
        current_pixel = stack.pop()
        x, y = current_pixel
        if 0 <= x < height and 0 <= y < width and visited[x,y] == 0:
            intensity_diff = abs(int(img[x,y]) - int(seed_intensity))

            if intensity_diff <= region_threshold:
                segmented_img[x,y] = 255
                visited[x,y] = 1
                for neighbour in neighbours:
                    neighbour_x = x + neighbour[0]
                    neighbour_y = y + neighbour[1]
                    stack.append((neighbour_x, neighbour_y))
    
    return segmented_img

def RBM_region_splitting(img, threshold):
    height, width = img.shape
    if height <= 1 or width <= 1:
        return img

    mean_value = np.mean(img)
    if mean_value <= threshold:
        return np.zeros_like(img)
    else:
        half_height = height // 2
        half_width = width // 2
        regions = []
        regions.append(img[:half_height, :half_width])
        regions.append(img[:half_height, half_width:])
        regions.append(img[half_height:, :half_width])
        regions.append(img[half_height:, half_width:])
        segmented_regions = [RBM_region_splitting(region, threshold) for region in regions]
        # Ensure all regions have the same dimensions before concatenating
        max_height = max(region.shape[0] for region in segmented_regions)
        max_width = max(region.shape[1] for region in segmented_regions)
        # Resize regions to match the maximum dimensions
        resized_regions = [np.pad(region, ((0, max_height - region.shape[0]), (0, max_width - region.shape[1])), 'constant') for region in segmented_regions]
        segmented_image = np.vstack([np.hstack([resized_regions[0], resized_regions[1]]),
                                     np.hstack([resized_regions[2], resized_regions[3]])])
        return segmented_image


#################### STREAMLIT APP ####################

# Defining the main function
def main():
    # Setting the title of the app

    st.set_page_config(
    page_title="Advanced Image Processor",
    page_icon="🖼️",
    layout="centered",
    initial_sidebar_state="expanded",
    )

    st.title("Advanced Image Processor")

    st.markdown("---")

    #################### SIDEBAR ####################
    st.sidebar.title("💡About")
    st.sidebar.subheader("This is a web app to perform and learn about image processing techniques on a given input image.")
    if st.sidebar.button("ℹ️ More Info"):
        switch_page("About and Techniques")

    st.sidebar.title("👨🏽‍💻Developer")
    st.sidebar.info(
        "This app is created by **Aadith Sukumar**\n"
        "\n🤝🏽[**Connect on LinkedIn**](https://www.linkedin.com/in/aadith-sukumar) \n"
        "\n👾[**Follow on GitHub**](https://www.github.com/aadi1011)\n"
    )
    ########################################
    
    uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file is not None:
        options = ["None", "Fourier Domain Filtering", "Edge Detection", "Image Segmentation", "Region-Based Methods"]
        choice = st.selectbox("Select the method you want to apply", options)

        # Checking if the user has selected a technique
        if choice == "None":
            st.warning("Please select a technique")

        elif choice == "Fourier Domain Filtering":
            technique = st.selectbox("Select the type of filter", ("Low Pass Filter", "High Pass Filter", "Blur Image", "Sharpen Image", "Gaussian Noise", "Salt and Pepper Noise"))
    
            if technique == "Low Pass Filter":
                D0 = st.slider("Select the cutoff frequency", 0, 100, 30)
                n = st.slider("Select the order of the filter", 1, 10, 2)
                img = np.array(Image.open(uploaded_file))
                img_filt = FDF_lowpass(img, D0, n)
                st.pyplot(plot_image(img_filt))
                try:
                    st.write("PSNR: ", PSNR(img, img_filt))
                except ValueError as ve:
                    st.warning(f"Could not calculate PSNR Value for this method")
                if st.checkbox("Show Histogram"):
                    st.pyplot(plot_histogram(img_filt))
            
            elif technique == "High Pass Filter":
                D0 = st.slider("Select the cutoff frequency", 0, 100, 30)
                n = st.slider("Select the order of the filter", 1, 10, 2)
                img = np.array(Image.open(uploaded_file))
                img_filt = FDF_highpass(img, D0, n)
                st.pyplot(plot_image(img_filt))
                try:
                    st.write("PSNR: ", PSNR(img, img_filt))
                except ValueError as ve:
                    st.warning(f"Could not calculate PSNR Value for this method")
                if st.checkbox("Show Histogram"):
                    st.pyplot(plot_histogram(img_filt))

            elif technique == "Blur Image":
                kernel_size = st.slider("Select the kernel size", 1, 100, 5)
                img = np.array(Image.open(uploaded_file))
                img_filt = FDF_blur(img, kernel_size)
                st.pyplot(plot_image(img_filt))
                try:
                    st.write("PSNR: ", PSNR(img, img_filt))
                except ValueError as ve:
                    st.warning(f"Could not calculate PSNR Value for this method")
                if st.checkbox("Show Histogram"):
                    st.pyplot(plot_histogram(img_filt))

            elif technique == "Sharpen Image":
                kernel_size = st.slider("Select the kernel size", 1, 100, 5)                
                img = np.array(Image.open(uploaded_file))
                img_filt = FDF_sharpen(img, kernel_size)
                st.pyplot(plot_image(img_filt))
                try:
                    st.write("PSNR: ", PSNR(img, img_filt))
                except ValueError as ve:
                    st.warning(f"Could not calculate PSNR Value for this method")
                if st.checkbox("Show Histogram"):
                    st.pyplot(plot_histogram(img_filt))

            elif technique == "Gaussian Noise":
                mean = st.slider("Select the mean", 0, 100, 0)
                stddev = st.slider("Select the standard deviation", 0, 100, 10)
                img = np.array(Image.open(uploaded_file))
                img_filt = FDF_gaussian_noise(img, mean, stddev)
                st.pyplot(plot_image(img_filt))
                try:
                    st.write("PSNR: ", PSNR(img, img_filt))
                except ValueError as ve:
                    st.warning(f"Could not calculate PSNR Value for this method")
                if st.checkbox("Show Histogram"):
                    st.pyplot(plot_histogram(img_filt))

            elif technique == "Salt and Pepper Noise":
                val = st.slider("Select the noise density", 0.0, 1.0, 0.5)
                img = np.array(Image.open(uploaded_file))
                img_filt = FDF_salt_pepper_noise(img, val)
                st.pyplot(plot_image(img_filt))
                try:
                    st.write("PSNR: ", PSNR(img, img_filt))
                except ValueError as ve:
                    st.warning(f"Could not calculate PSNR Value for this method")
                if st.checkbox("Show Histogram"):
                    st.pyplot(plot_histogram(img_filt))

        elif choice == "Edge Detection":
            technique = st.selectbox("Select the type of filter", ("Sobel Filter", "Prewitt Filter", "Roberts Filter", "Canny Edge Detection"))
        
            if technique == "Sobel Filter":
                kernel_size = st.slider("Select the kernel size", 1, 31, 5, step=2)
                img = np.array(Image.open(uploaded_file))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                
                img_filt = ED_sobel(img, kernel_size)
                st.pyplot(plot_image(img_filt))
                try:
                    st.write("PSNR: ", PSNR(img, img_filt))
                except ValueError as ve:
                    st.warning(f"Could not calculate PSNR Value for this method")
                if st.checkbox("Show Histogram"):
                    st.pyplot(plot_histogram(img_filt))

            elif technique == "Prewitt Filter":
                img = np.array(Image.open(uploaded_file))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                
                img_filt = ED_prewitt(img)
                st.pyplot(plot_image(img_filt))
                try:
                    st.write("PSNR: ", PSNR(img, img_filt))
                except ValueError as ve:
                    st.warning(f"Could not calculate PSNR Value for this method")
                if st.checkbox("Show Histogram"):
                    st.pyplot(plot_histogram(img_filt))

            elif technique == "Roberts Filter":
                img = np.array(Image.open(uploaded_file))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                                
                img_filt = ED_roberts(img)
                st.pyplot(plot_image(img_filt))
                try:
                    st.write("PSNR: ", PSNR(img, img_filt))
                except ValueError as ve:
                    st.warning(f"Could not calculate PSNR Value for this method")
                if st.checkbox("Show Histogram"):
                    st.pyplot(plot_histogram(img_filt))

            elif technique == "Canny Edge Detection":
                upper_threshold = st.slider("Select the upper threshold", 0, 255, 100)
                lower_threshold = st.slider("Select the lower threshold", 0, 255, 50)
                img = np.array(Image.open(uploaded_file))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                                
                img_filt = ED_canny(img, upper_threshold, lower_threshold)
                st.pyplot(plot_image(img_filt))
                try:
                    st.write("PSNR: ", PSNR(img, img_filt))
                except ValueError as ve:
                    st.warning(f"Could not calculate PSNR Value for this method")
                if st.checkbox("Show Histogram"):
                    st.pyplot(plot_histogram(img_filt))

        elif choice == "Image Segmentation":
            technique = st.selectbox("Select the type of filter", ("Binary Thresholding", "Inverse Binary Thresholding", "Truncated Thresholding", "To Zero Thresholding", "Otsu Thresholding", "Gaussian Thresholding", "Mean Adaptive Thresholding", "Gaussian Adaptive Thresholding"))

            if technique == "Binary Thresholding":
                threshold = st.slider("Select the threshold", 0, 255, 100)
                img = np.array(Image.open(uploaded_file))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                                
                img_filt = IS_binary_thresholding(img, threshold)
                st.pyplot(plot_image(img_filt))
                try:
                    st.write("PSNR: ", PSNR(img, img_filt))
                except ValueError as ve:
                    st.warning(f"Could not calculate PSNR Value for this method")
                if st.checkbox("Show Histogram"):
                    st.pyplot(plot_histogram(img_filt))

            elif technique == "Inverse Binary Thresholding":
                threshold = st.slider("Select the threshold", 0, 255, 100)
                img = np.array(Image.open(uploaded_file))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                                
                img_filt = IS_inverse_binary_thresholding(img, threshold)
                st.pyplot(plot_image(img_filt))
                try:
                    st.write("PSNR: ", PSNR(img, img_filt))
                except ValueError as ve:
                    st.warning(f"Could not calculate PSNR Value for this method")
                if st.checkbox("Show Histogram"):
                    st.pyplot(plot_histogram(img_filt))

            elif technique == "Truncated Thresholding":
                threshold = st.slider("Select the threshold", 0, 255, 100)
                img = np.array(Image.open(uploaded_file))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                                
                img_filt = IS_truncated_thresholding(img, threshold)
                st.pyplot(plot_image(img_filt))
                try:
                    st.write("PSNR: ", PSNR(img, img_filt))
                except ValueError as ve:
                    st.warning(f"Could not calculate PSNR Value for this method")
                if st.checkbox("Show Histogram"):
                    st.pyplot(plot_histogram(img_filt))

            elif technique == "To Zero Thresholding":
                threshold = st.slider("Select the threshold", 0, 255, 100)
                img = np.array(Image.open(uploaded_file))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                                
                img_filt = IS_to_zero_thresholding(img, threshold)
                st.pyplot(plot_image(img_filt))
                try:
                    st.write("PSNR: ", PSNR(img, img_filt))
                except ValueError as ve:
                    st.warning(f"Could not calculate PSNR Value for this method")
                if st.checkbox("Show Histogram"):
                    st.pyplot(plot_histogram(img_filt))

            elif technique == "Otsu Thresholding":
                img = np.array(Image.open(uploaded_file))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                                
                img_filt = IS_otsu_thresholding(img)
                st.pyplot(plot_image(img_filt))
                try:
                    st.write("PSNR: ", PSNR(img, img_filt))
                except ValueError as ve:
                    st.warning(f"Could not calculate PSNR Value for this method")
                if st.checkbox("Show Histogram"):
                    st.pyplot(plot_histogram(img_filt))

            elif technique == "Gaussian Thresholding":
                img = np.array(Image.open(uploaded_file))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                                
                img_filt = IS_gaussian_thresholding(img)
                st.pyplot(plot_image(img_filt))
                try:
                    st.write("PSNR: ", PSNR(img, img_filt))
                except ValueError as ve:
                    st.warning(f"Could not calculate PSNR Value for this method")
                if st.checkbox("Show Histogram"):
                    st.pyplot(plot_histogram(img_filt))

            elif technique == "Mean Adaptive Thresholding":
                block_size = st.slider("Select the block size", 3, 255, 5, step=2)
                constant = st.slider("Select the constant", 0, 255, 5)
                img = np.array(Image.open(uploaded_file))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                                
                img_filt = IS_mean_adaptive_thresholding(img, block_size, constant)
                st.pyplot(plot_image(img_filt))
                try:
                    st.write("PSNR: ", PSNR(img, img_filt))
                except ValueError as ve:
                    st.warning(f"Could not calculate PSNR Value for this method")
                if st.checkbox("Show Histogram"):
                    st.pyplot(plot_histogram(img_filt))

            elif technique == "Gaussian Adaptive Thresholding":
                block_size = st.slider("Select the block size", 3, 255, 5, step=2)
                constant = st.slider("Select the constant", 0, 255, 5)
                img = np.array(Image.open(uploaded_file))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                                
                img_filt = IS_gaussian_adaptive_thresholding(img, block_size, constant)
                st.pyplot(plot_image(img_filt))
                try:
                    st.write("PSNR: ", PSNR(img, img_filt))
                except ValueError as ve:
                    st.warning(f"Could not calculate PSNR Value for this method")
                if st.checkbox("Show Histogram"):
                    st.pyplot(plot_histogram(img_filt))

        elif choice == "Region-Based Methods":
            technique = st.selectbox("Select the type of filter", ("Region Growing", "Region Splitting"))

            if technique == "Region Growing":
                threshold = st.slider("Select the threshold", 0, 255, 100)
                img = np.array(Image.open(uploaded_file))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                height, width = img.shape
                seed = (height//2, width//2)
                img_filt = RBM_region_growing(img, seed, threshold)
                st.pyplot(plot_image(img_filt))
                try:
                    st.write("PSNR: ", PSNR(img, img_filt))
                except ValueError as ve:
                    st.warning(f"Could not calculate PSNR Value for this method")
                if st.checkbox("Show Histogram"):
                    st.pyplot(plot_histogram(img_filt))

            elif technique == "Region Splitting":
                threshold = st.slider("Select the threshold", 0, 255, 100)
                img = np.array(Image.open(uploaded_file))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_filt = RBM_region_splitting(img, threshold)
                st.pyplot(plot_image(img_filt))
                try:
                    st.write("PSNR: ", PSNR(img, img_filt))
                except ValueError as ve:
                    st.warning(f"Could not calculate PSNR Value for this method")
                if st.checkbox("Show Histogram"):
                    st.pyplot(plot_histogram(img_filt))


if __name__ == "__main__":  
    main()
