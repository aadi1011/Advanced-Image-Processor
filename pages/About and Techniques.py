import streamlit as st
from streamlit_extras.switch_page_button import switch_page 


About_Project_Text = """
Welcome to my fascinating project, where we embark on a comprehensive exploration of diverse image processing techniques, unraveling the science of digital images through the lens of digital transformations. Here's a glimpse of the key highlights:

In the realm of image processing, we delved into the intricacies of Fourier Domain Transform, leveraging Butterworth filters for frequency domain manipulation, and Gaussian filters for image blurring. The arsenal of edge detection methods, including Sobel, Prewitt, Roberts, and the advanced Canny edge detection, empowered us to pinpoint object boundaries with precision. Addressing challenges such as Gaussian and Salt-and-Pepper noise, we adeptly applied tailored filters to restore the pristine clarity of our images.

Our journey extended into the realm of image segmentation, embracing binary, truncated, to zero, Otsu's, and adaptive thresholding methods. Moreover, we explored region growing, splitting, and merging algorithms, providing a panoramic view of segmentation strategies. Applying these techniques to various images showcased the tangible impact of image processing.

The practical utility of our efforts became evident as we mitigated noise, sharpened edges, and enhanced the interpretability of the images. Utilizing metrics like Peak Signal-to-Noise Ratio (PSNR), we ensured not only visual appeal but also statistical significance in the quality assessment of our processed images.

The project's challenges and learnings were profound, particularly in parameter selection for segmentation techniques. Navigating the interplay between techniques and image nuances demanded a deep understanding, while grounding our work in the real-world context proved crucial. Adapting generic techniques to any given image required ingenuity and critical thinking.

Looking ahead, our project envisions exciting prospects, including the integration of Deep Learning. Exploring the potential of models like convolutional neural networks (CNNs) could usher in significant advancements for complex image analysis tasks. Additionally, we foresee Interactive Visualization as a future consideration, envisioning interactive tools that allow users to explore segmented regions and apply filters dynamically.

In essence, our project underscores the profound power and versatility of image processing techniques and visual data analysis. Through meticulous implementation and creative problem-solving, we have successfully elevated the quality and interpretability of any given image, emphasizing the expansive potential of image processing across diverse fields.

\- [**Aadith Sukumar**](www.linkedin.com/in/aadith-sukumar)
"""

Fourier_Domain_Text = """
Fourier Domain Transform is a fundamental concept in image processing that involves transforming an image from spatial domain to frequency domain using Fourier analysis. In this domain, images are represented as a sum of sinusoidal functions of varying frequencies and amplitudes. The transformation is performed using the Fast Fourier Transform (FFT) algorithm, which efficiently computes the Fourier coefficients. This transformation provides valuable insights into the frequency components of an image, highlighting patterns, edges, and textures.

The Fourier domain allows the separation of image information based on its frequency content. High-frequency components correspond to rapid changes or edges in the image, while low-frequency components represent smooth areas. Applications of Fourier Domain Transform include image filtering, compression, and analysis. Filtering in the frequency domain involves modifying specific frequency components, enabling tasks like noise removal and sharpening. Additionally, Fourier analysis finds applications in fields such as medical imaging, astronomy, and signal processing. The mathematical representation of 2D Fourier Transform is given by:

$$ (u, v) = \iint f(x, y) \cdot e^{-2\pi i (ux + vy)} \, dx \, dy $$

Where $F(u, v)$ represents the Fourier transform of the image $f(x, y)$, and $u$ and $v$ are the spatial frequencies.
"""

Butterworth_Filter_Text = """
The Butterworth filter is a type of frequency domain filter used in image processing for tasks like image enhancement, noise reduction, and edge detection. Unlike ideal filters, Butterworth filters have a smoother transition between the passband and stopband, minimizing abrupt changes in the frequency response. The filter operates in the frequency domain, modifying the frequency components of an image.

The Butterworth filter transfer function in the frequency domain is given by the equation:

$$ H(u, v) = \\frac{1}{1 + \left(\\frac{D(u, v)}{D_0}\\right)^{2n}} $$ 

Where:
- $ H(u, v) $ is the filter transfer function.
- $ D(u, v) $ represents the distance from a point \((u, v)\) in the frequency domain to the center.
- $ D_0 $ is the cutoff frequency, determining the frequency beyond which the filtering occurs.
- $ n $ is the order of the filter, controlling the sharpness of the transition between passband and stopband.

**Applications:**

1. **Image Enhancement:** Butterworth filters can be used to enhance specific frequency components, highlighting fine details in an image.
2. **Noise Reduction:** By attenuating high-frequency noise components, Butterworth filters are effective in reducing noise in images.
3. **Edge Detection:** Butterworth high-pass filters accentuate edges and boundaries in images by amplifying high-frequency components.
"""

Blur_Text = """
**Blur:**
Blur in image processing refers to the reduction of sharpness or detail in an image. It's achieved by averaging the pixel values in a neighborhood around each pixel. The resulting pixel value is a weighted average of the neighboring pixels, which reduces high-frequency noise and smoothens the image. Blurring is commonly used in applications like image denoising and simplification.

**Gaussian Blur:**
Gaussian blur is a specific type of blur that uses a Gaussian function to calculate the weights of neighboring pixels. In this method, the weights decrease as the distance from the center pixel increases, following the bell-shaped Gaussian curve. This type of blur produces a smoother and more natural-looking result compared to simple averaging. It's widely used in image processing due to its effectiveness and ability to preserve edges while reducing noise.

**Median Blur:**
Median blur is a non-linear filter that replaces the pixel value in the center of the neighborhood with the median value of all the pixels in that region. Unlike linear filters, median blur does not calculate a weighted average, making it robust against outliers and impulse noise. Median blur is particularly useful for removing salt-and-pepper noise, a type of noise characterized by randomly occurring white and black pixels in an image.

**Applications:**
- **Blur:** Used in artistic effects, privacy protection by obscuring sensitive information, and reducing noise in images.
- **Gaussian Blur:** Commonly employed in image smoothing, noise reduction, and as a preprocessing step in edge detection algorithms.
- **Median Blur:** Effective for removing impulsive noise in medical imaging, satellite imagery, and any application where preserving edges is crucial.
"""

Sharpen_Text = """
**Image Sharpening** is a technique in image processing that enhances the edges and fine details in an image, making it appear clearer and more defined. The process involves emphasizing the high-frequency components of an image, which correspond to abrupt changes in intensity, such as edges. Image sharpening techniques work by accentuating these high-frequency components while preserving the overall structure and details of the image.

**Applications:**

- Photography: Image sharpening is widely used in photography to enhance details and make images appear crisp and clear.
- Medical Imaging: Sharpening is applied to medical images to highlight fine structures, aiding in diagnosis.
- Industrial Inspection: In applications like quality control, sharpening helps detect imperfections in manufactured products.
- Satellite Imaging: Sharpening techniques improve the clarity of satellite images, enhancing the visibility of features on the Earth's surface.
"""

Gaussian_Noise_Text = """
Gaussian noise is a type of statistical noise with a probability density function (PDF) that follows a Gaussian (normal) distribution. In images, Gaussian noise appears as random variations in pixel values, where each value is independently drawn from a Gaussian distribution with mean
$μ$ and standard deviation $σ$. This noise type is prevalent in various natural and man-made processes, such as electronic sensors or transmission over communication channels.

Low pass filters are essential tools in image processing that allow low-frequency components (such as smooth areas and gradual transitions) to pass through while attenuating high-frequency components (such as noise and fine details). The filtering process smoothens the image, reducing noise and emphasizing broader features. Different kernel sizes in low pass filters control the extent of smoothing: larger kernels result in more extensive smoothing, while smaller kernels preserve finer details.

**Applications:**

- Image Smoothing: Low pass filters are commonly used to remove noise, resulting in a smoother image suitable for further processing or analysis.
- Preprocessing for High Pass Filters: Low pass filtering is often a preliminary step before applying high pass filters for tasks like edge detection.
- Image Compression: In image compression algorithms, smoothing reduces high-frequency noise, enhancing the efficiency of compression algorithms like JPEG.
"""

Salt_Pepper_Noise_Text = """
Salt and pepper noise is a type of digital noise that appears as randomly occurring white and black pixels in an image. This noise is caused by errors during image acquisition or transmission, resulting in sudden, random spikes of intensity. Salt and pepper noise can severely degrade the quality of an image, making it challenging to interpret and process. It gets its name because the white pixels resemble grains of salt, while the black pixels resemble grains of pepper. This noise occurs due to various factors during image acquisition, transmission, or storage, such as errors in sensors or communication channels.

In image processing, each pixel's intensity values range from 0 (black) to 255 (white) in an 8-bit grayscale image. Salt and pepper noise corrupts these intensity values by setting some pixels to the minimum intensity (0) and others to the maximum intensity (255), randomly and independently. Dealing with salt and pepper noise is essential in image processing because it can significantly affect the image's visual quality and distort important features. Techniques such as median filtering and adaptive median filtering are commonly used to remove salt and pepper noise by replacing the noisy pixel values with the median values of neighboring pixels, effectively smoothing out the image while preserving its edges and important structures.
"""

Edge_Detection_Text = """
Edge detection is a fundamental concept in image processing that aims to identify boundaries within an image, specifically areas where pixel intensity significantly changes. These boundaries often represent important features in the image, such as object boundaries or transitions between different textures. Edge detection plays a crucial role in various image analysis tasks, including object recognition, image segmentation, and computer vision applications.

**Techniques for Edge Detection:**

- Sobel Operator
- Prewitt Operator
- Roberts Cross Operator
- Canny Edge Detector

**Applications of Edge Detection:**

- Object Recognition: Edge detection helps identify objects in images by outlining their boundaries.
- Image Segmentation: Detecting edges assists in partitioning an image into meaningful segments.
- Robotics and Autonomous Vehicles: Edge information is vital for navigation and obstacle avoidance.
- Medical Imaging: Edge detection aids in detecting boundaries of organs and structures in medical images.

**Challenges in Edge Detection:**

- Noise Sensitivity: Edge detection algorithms are sensitive to image noise. Pre-processing steps like smoothing are often applied to mitigate this issue.
- Parameter Tuning: Many edge detection methods have parameters that need to be tuned based on the characteristics of the image and the application, making the process somewhat subjective.
"""

Sobel_Text = """
Sobel edge detection is a popular gradient-based method used for edge detection in image processing. It calculates the approximate absolute gradient of the image intensity function, highlighting areas of rapid intensity change. Sobel operators are particularly effective in detecting edges in different orientations.


For a grayscale image, Sobel edge detection involves convolving the image with two 3x3 kernels: one for detecting edges in the horizontal direction $(G_x)$ and another for the vertical direction $(G_y)$. These kernels are designed to approximate the derivative of the image intensity function. For $G_x$:

$ G_x = \\begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix} $

For $G_y$:

$ G_y = \\begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix} $

To compute the gradient magnitude $(G)$ at each pixel, the following formula is used:

$ G = \sqrt{G_x^2 + G_y^2} $

And the gradient direction $(\\theta)$ can be calculated as:

$ \\theta = \\arctan\left(\\frac{G_y}{G_x}\\right) $

Where:
- $G_x$ and $G_y$ represent the convolutions of the image with the Sobel kernels for the horizontal and vertical directions, respectively.
- $G$ is the gradient magnitude, representing the strength of the edge.
- $\\theta$ is the gradient direction, indicating the orientation of the edge.

After obtaining $G$ and $\\theta$, non-maximum suppression is often applied to thin the edges, followed by hysteresis thresholding to link and finalize the detected edges.

Sobel edge detection is widely used due to its simplicity and effectiveness in detecting edges in different directions. It is a key step in many edge detection algorithms and computer vision applications.
"""

Prewitt_Text = """
Prewitt edge detection is another gradient-based method used for edge detection in image processing, similar to Sobel operators. It is designed to approximate the derivative of the image intensity function, emphasizing areas of rapid intensity change, and is particularly effective for detecting edges in different orientations.

For Prewitt edge detection, two 3x3 kernels are used: one for detecting edges in the horizontal direction $(P_x)$ and another for the vertical direction $(P_y)$. These kernels are designed to calculate the first derivative of the image intensity function. For $P_x$:

$ P_x = \\begin{bmatrix} -1 & 0 & 1 \\ -1 & 0 & 1 \\ -1 & 0 & 1 \end{bmatrix} $

For $P_y$:

$ P_y = \\begin{bmatrix} -1 & -1 & -1 \\ 0 & 0 & 0 \\ 1 & 1 & 1 \end{bmatrix} $

To compute the gradient magnitude $(G)$ at each pixel, the following formula is used:

$ G = \sqrt{P_x^2 + P_y^2} $

And the gradient direction $(\\theta)$ can be calculated as:

$ \\theta = \\arctan\left(\\frac{P_y}{P_x}\\right) $

Where:
- $P_x$ and $P_y$ represent the convolutions of the image with the Prewitt kernels for the horizontal and vertical directions, respectively.
- $G$ is the gradient magnitude, representing the strength of the edge.
- $\\theta$ is the gradient direction, indicating the orientation of the edge.

After obtaining $G$ and $\\theta$, similar to Sobel operators, non-maximum suppression and hysteresis thresholding can be applied to identify and refine the detected edges.

Prewitt edge detection is widely used in various image processing applications, especially when detecting edges in multiple directions is necessary. It is computationally efficient and straightforward to implement, making it a popular choice in real-time applications and computer vision systems.
"""

Roberts_Text = """
Roberts edge detection is one of the simplest gradient-based methods used for edge detection in image processing. Like Sobel and Prewitt operators, Roberts operators focus on detecting edges by emphasizing areas of rapid intensity change. However, Roberts operators use a pair of 2x2 kernels to approximate the gradient of the image intensity function.

For Roberts edge detection, two 2x2 kernels are used: one for detecting edges in the horizontal direction $(R_x)$ and another for the vertical direction $(R_y)$. These kernels are designed to calculate the first derivative of the image intensity function. For $(R_x)$:

$ R_x = \\begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix} $

For $(R_y$):

$ R_y = \\begin{bmatrix} 0 & 1 \\ -1 & 0 \end{bmatrix} $

To compute the gradient magnitude $(G$) at each pixel, the following formula is used:

$ G = \sqrt{R_x^2 + R_y^2} $

And the gradient direction $(\\theta$) can be calculated as:

$ \\theta = \\arctan\left(\\frac{R_y}{R_x}\\right) $

Where:
- $R_x$ and $R_y$ represent the convolutions of the image with the Roberts kernels for the horizontal and vertical directions, respectively.
- $G$ is the gradient magnitude, representing the strength of the edge.
- $\\theta$ is the gradient direction, indicating the orientation of the edge.

After obtaining $G$ and $\\theta$, non-maximum suppression and hysteresis thresholding can be applied to identify and refine the detected edges, similar to other gradient-based edge detection methods.

Roberts edge detection is computationally lightweight due to its small kernels, making it suitable for real-time applications and scenarios where computational resources are limited. However, it is more sensitive to noise compared to some other edge detection techniques.
"""

Canny_Text = """
Canny edge detection is a multi-step algorithm widely used for edge detection in image processing. It aims to identify edges by detecting areas of rapid intensity change, providing precise edge localization and good noise tolerance.

**Steps:**

1. **Gaussian Smoothing:**
   Canny edge detection starts by applying a Gaussian filter to the image to reduce noise and eliminate fine details.

2. **Gradient Calculation:**
   Sobel operators are used to compute the image gradients $(G_x$) and $(G_y$) in horizontal and vertical directions. The gradient magnitude $(G$) and direction $(\\theta$) are calculated using these gradients.

   $ G = \sqrt{G_x^2 + G_y^2} $
   $ \\theta = \\arctan\left(\\frac{G_y}{G_x}\\right) $

3. **Non-Maximum Suppression:**
   Non-maximum suppression is performed to thin the edges by preserving only the local maxima in the gradient direction. Only pixels with gradient magnitudes larger than their neighbors in the direction of the gradient are kept.

4. **Double Thresholding:**
   Two thresholds, high and low, are applied to the gradient magnitudes. Pixels with gradient magnitudes higher than the high threshold are considered strong edges, while pixels between the high and low thresholds are considered weak edges.

5. **Edge Tracking by Hysteresis:**
   Weak edges are connected to strong edges if they are part of the same edge structure. This step involves tracking weak edges adjacent to strong edges. If a weak edge pixel is connected to a strong edge pixel, it is considered part of the edge.

**Equations:**

- Gradient Magnitude:
  $ G = \sqrt{G_x^2 + G_y^2} $

- Gradient Direction:
  $ \\theta = \\arctan\left(\\frac{G_y}{G_x}\\right) $

Where:
- $G_x$ and $G_y$ are the gradients in the horizontal and vertical directions, respectively.

Canny edge detection is known for its ability to provide accurate edge maps and is widely used in computer vision applications, object recognition, and image analysis tasks.
"""

PSNR_Text = """
**Peak Signal-to-Noise Ratio for Image Segmentation:**

Peak Signal-to-Noise Ratio (PSNR) is a metric used to evaluate the quality of an image after it has undergone compression or processing, such as image segmentation. It measures the fidelity of the processed image concerning the original, providing insights into the level of information preservation.

**Equation:**

The PSNR is calculated using the following formula:

$$ PSNR = 10 \cdot \log_{10}\left(\\frac{{\\text{MAX}^2}}{{\\text{MSE}}}\\right) $$

Where:
- $ \\text{MAX}$ represents the maximum possible pixel value in the image (typically 255 for an 8-bit grayscale image).
- $\\text{MSE}$ is the Mean Squared Error between the original and processed images. It is calculated as the average of the squared differences between corresponding pixels in the two images.

**Application in Image Segmentation:**

In the context of image segmentation, PSNR can be utilized to assess the accuracy and quality of the segmented image compared to the ground truth or manually segmented image. Higher PSNR values indicate lower distortion between the original and segmented images, implying better segmentation results.

⚠️***Note:** While PSNR is widely used, it's important to note that it has limitations, especially in the context of perceptual image quality. It does not always align with human visual perception, as it assumes that all pixel errors are equally visible, which might not be the case in all practical scenarios.*
"""

Binary_Thresholding_Text = """
**Binary Thresholding and Inverse Binary Thresholding for Image Segmentation:**

Binary thresholding and inverse binary thresholding are simple yet effective techniques used in image segmentation. These methods divide an image into two regions based on pixel intensity values, categorizing pixels into either foreground or background, making them valuable for tasks like object detection and extraction.

**Binary Thresholding:**

In binary thresholding, pixels with intensity values above a specified threshold are set to one (foreground), while pixels below the threshold are set to zero (background). The thresholding operation is mathematically defined as follows:

$$ \\text{Binary Thresholding:} \quad
\\text{Output}(x, y) = \\begin{cases}
1, & \\text{if } \\text{Input}(x, y) > \\text{Threshold} \\
0, & \\text{otherwise}
\end{cases}
$$

**Inverse Binary Thresholding:**

Inverse binary thresholding, on the other hand, flips the logic. Pixels with intensity values above the threshold are set to zero (background), while pixels below the threshold are set to one (foreground):

$$ \\text{Inverse Binary Thresholding:} \quad
\\text{Output}(x, y) = \\begin{cases}
0, & \\text{if } \\text{Input}(x, y) > \\text{Threshold} \\
1, & \\text{otherwise}
\end{cases}
$$

**Applications:**

- **Object Detection:** Binary thresholding is commonly used to separate objects from the background in images.
- **Image Enhancement:** By isolating specific regions based on intensity, these techniques help enhance particular features in an image.
- **Document Processing:** Used in OCR (Optical Character Recognition) tasks to distinguish text from the background.

**Considerations/Challenges:**

- **Threshold Selection:** Choosing an appropriate threshold value is critical. Various techniques like Otsu's method can be employed for automatic threshold selection.
- **Noise Sensitivity:** These methods are sensitive to noise. Pre-processing steps such as blurring can be applied to mitigate the effects of noise before thresholding.

Binary and inverse binary thresholding provide a straightforward approach to image segmentation, making them valuable tools in many image processing applications.
"""

Truncated_Thresholding_Text = """
**Truncated Thresholding for Image Segmentation:**

Truncated thresholding is a technique used in image segmentation, where pixels with intensity values above a specified threshold are set to the threshold value, and pixels below the threshold remain unchanged. This method effectively transforms the image into a binary image with a limited dynamic range.


$$ \\text{Truncated Thresholding:} \quad
\\text{Output}(x, y) = \\begin{cases}
\\text{Threshold}, & \\text{if } \\text{Input}(x, y) > \\text{Threshold} \\
\\text{Input}(x, y), & \\text{otherwise}
\end{cases}
$$

**Applications:**

- **Image Simplification:** Truncated thresholding simplifies the image by reducing the number of intensity levels, making it easier to process and analyze.
- **Feature Extraction:** By emphasizing specific intensity values, this method helps extract certain features from an image.
- **Contrast Enhancement:** Truncated thresholding can be used to enhance the contrast of specific regions in the image.

**Considerations/Challenges:**

- **Threshold Selection:** Choosing an appropriate threshold value is crucial. It often depends on the characteristics of the image and the specific segmentation task.
- **Loss of Information:** Truncated thresholding results in the loss of intensity information above the threshold, which may impact the quality of the segmented image. Careful consideration is needed when choosing the threshold to minimize information loss.

Truncated thresholding provides a straightforward way to segment images by simplifying their intensity values. While it may result in information loss, it is a useful technique in scenarios where simplifying the image's intensity levels is acceptable and beneficial for the specific application.

"""

To_Zero_Thresholding_Text = """
**To Zero Thresholding for Image Segmentation:**

To Zero thresholding, also known as zeroing thresholding, is a technique used in image processing for segmentation. In this method, pixels with intensity values below a specified threshold are set to zero, while pixels with values equal to or above the threshold remain unchanged.

**Equation:**

$$ \\text{To Zero Thresholding:} \quad
\\text{Output}(x, y) = \\begin{cases}
\\text{Input}(x, y), & \\text{if } \\text{Input}(x, y) \geq \\text{Threshold} \\
0, & \\text{otherwise}
\end{cases}
$$

**Applications:**

- **Object Detection:** To Zero thresholding helps in separating objects from the background in an image by setting the background pixels to zero.
- **Noise Reduction:** It can be used to suppress noise in images by setting low-intensity noise pixels to zero, assuming the objects of interest have higher intensities.
- **Image Masking:** To Zero thresholding is used to create masks where regions of interest are preserved, and other areas are set to zero.

**Considerations:**

- **Threshold Selection:** Choosing an appropriate threshold value is critical and often depends on the specific characteristics of the image and the segmentation requirements.
- **Information Loss:** Pixels below the threshold are set to zero, leading to potential loss of information in the resulting segmented image. This is especially important in applications where preserving all intensity levels is crucial.

To Zero thresholding provides a simple way to segment images by emphasizing specific intensity values and setting others to zero. It is widely used in various image analysis tasks where a binary representation of the objects in the image is sufficient for further processing or analysis.
"""

Otsu_Thresholding_Text = """
**Otsu's Thresholding for Image Segmentation:**

Otsu's Thresholding, also known as Otsu's method or maximum variance method, is an automatic thresholding technique used in image processing for segmentation. It calculates an optimal threshold value by maximizing the interclass variance between the background and foreground pixels in the image, effectively separating objects from the background.

**Algorithm:**

1. **Compute Histogram:** Calculate the histogram of the input image to determine the frequency of each intensity level.

2. **Compute Cumulative Distribution:** Compute the cumulative distribution function (CDF) and the cumulative sum of intensities.

3. **Calculate Total Variance:** Calculate the total variance $( \sigma^2_{\\text{total}}) $ of the image intensity values.

4. **Iterate Through Thresholds:**
   - For each possible threshold value ($ t $):
     - Compute the weighted within-class variance ($ \sigma^2_{\\text{within}} $) for pixels below and above the threshold.
     - Calculate the between-class variance ($ \sigma^2_{\\text{between}} $) using the formula: 
       $$ \sigma^2_{\\text{between}} = \sigma^2_{\\text{total}} - \sigma^2_{\\text{within}} $$
     - Choose the threshold that maximizes $ \sigma^2_{\\text{between}} $.

5. **Apply Threshold:** Set pixels with intensity values above the optimal threshold to foreground and below or equal to the threshold to background.

**Applications:**

- **Object Segmentation:** Otsu's thresholding is widely used for segmenting objects from the background in various images.
- **Biomedical Imaging:** It's used in tasks like cell counting and tissue segmentation in medical images.
- **Document Analysis:** Otsu's method helps in separating text from the background in document images.

**Advantages:**

- **Automatic:** Otsu's method automatically determines the optimal threshold, reducing the need for manual threshold selection.
- **Mathematically Sound:** It is based on maximizing the variance, making it statistically robust.

**Considerations/Challenges:**

- **Bimodal Assumption:** Otsu's method assumes that the intensity histogram of the image has a bimodal distribution, meaning it works best for images with clear foreground and background intensity separations.

Otsu's Thresholding provides an efficient way to perform image segmentation, especially in scenarios where manual threshold selection can be challenging or time-consuming.
"""

Gaussian_Thresholding_Text = """
**Gaussian Thresholding for Image Segmentation:**

Gaussian Thresholding is a technique used in image processing for segmentation, where the threshold value is determined using a Gaussian distribution model fitted to the pixel intensities of the image. This method assumes that the pixel intensities in the image follow a Gaussian distribution and sets the threshold based on this statistical model.

**Algorithm:**

1. **Compute Histogram:** Calculate the histogram of the input image to determine the frequency of each intensity level.

2. **Fit Gaussian Distribution:** Estimate the parameters of a Gaussian distribution (mean and standard deviation) based on the pixel intensities from the image's histogram.

3. **Calculate Threshold:** Set the threshold to a specific number of standard deviations ($k$) away from the mean of the Gaussian distribution. The threshold is calculated as follows:
   $$ \\text{Threshold} = \\text{mean} + k \\times \\text{standard deviation} $$

4. **Apply Threshold:** Pixels with intensity values above the calculated threshold are classified as foreground, and those below or equal to the threshold are classified as background.

**Applications:**

- **Image Enhancement:** Gaussian Thresholding is used to enhance specific features in an image by isolating pixels with desired intensity levels.
- **Document Analysis:** It helps in segmenting text or handwritten characters from the background in document images.
- **Biomedical Imaging:** Gaussian Thresholding is applied in medical image analysis for tasks like tumor detection and cell segmentation.

**Advantages:**

- **Adaptability:** Gaussian Thresholding adapts to the intensity distribution of the image, making it suitable for images with varying lighting conditions and contrasts.
- **Statistical Basis:** It uses statistical properties of the image intensities, making it robust in handling noise and varying illumination.

**Considerations:**

- **Parameter Selection:** Choosing an appropriate value for $k$ (number of standard deviations) is essential and might require experimentation or domain knowledge.
- **Assumption of Gaussian Distribution:** The method assumes that the pixel intensities follow a Gaussian distribution, which might not always hold true for all types of images.

Gaussian Thresholding provides a robust and adaptable approach to image segmentation, making it valuable in scenarios where the intensity distribution of the image is a crucial factor in the segmentation process.
"""

Adaptive_Thresholding_Text = """
**Adaptive Thresholding for Image Segmentation:**

Adaptive Thresholding is a dynamic thresholding technique used in image processing for segmentation, especially in cases where the lighting conditions across an image vary significantly. Unlike global thresholding methods where a single threshold is applied to the entire image, adaptive thresholding calculates local thresholds for different regions of the image. This approach allows for better adaptability to local intensity variations.

**Algorithm:**

1. **Divide the Image into Tiles:** The input image is divided into small non-overlapping tiles or blocks.

2. **Calculate Local Thresholds:** For each tile, a local threshold is calculated using a specified method (commonly mean or Gaussian) considering only the pixel intensities within that tile.

3. **Apply Threshold Locally:** The local threshold obtained for each tile is applied to the pixels within that tile. Pixels with intensity values above the local threshold are classified as foreground, and those below or equal to the threshold are classified as background.

**Applications:**

- **Document Scanning:** Adaptive Thresholding is used to extract text and graphics from documents, even in the presence of varying lighting conditions and paper quality.
- **Image Binarization:** It's applied in preprocessing steps for OCR (Optical Character Recognition) tasks to create binary images for character segmentation.
- **Machine Vision:** Used in industrial applications for quality control and defect detection in varying lighting environments.

**Advantages:**

- **Robustness:** Adaptive Thresholding adapts to local intensity variations, making it robust in handling images with uneven lighting or contrast.
- **Improved Accuracy:** By considering local context, it provides more accurate segmentation results compared to global thresholding methods, especially for complex scenes.

**Considerations:**

- **Tile Size:** The choice of tile size is critical. Small tiles might capture noise, while large tiles might oversmooth important details. The tile size should be selected based on the characteristics of the image and the segmentation task.
- **Computational Complexity:** Adaptive Thresholding can be computationally intensive, especially for large images, due to the need to calculate thresholds for multiple tiles.

Adaptive Thresholding offers an effective solution for image segmentation in scenarios where lighting conditions vary across the image, ensuring accurate and reliable separation of foreground and background elements.
"""

Region_Growing_Text = """
**Region Growing Method for Image Segmentation:**

Region Growing is a seed-based image segmentation technique used in image processing to group pixels or voxels that have similar properties, such as intensity or color. The method starts with one or more seed points and grows a region by adding neighboring pixels that meet specific criteria, forming a connected and homogeneous region.

**Algorithm:**

1. **Seed Selection:** Choose one or more seed points in the image. These seeds can be manually selected or determined automatically based on certain criteria.

2. **Pixel Similarity Criterion:** Define a similarity criterion, often based on intensity or color, to determine whether a pixel can be added to the region. For example, a pixel may be added if its intensity is within a certain range of the seed point's intensity.

3. **Neighbor Expansion:** For each pixel in the current region, examine its neighbors. If a neighbor pixel satisfies the similarity criterion, add it to the region. This process continues iteratively, expanding the region.

4. **Stopping Criteria:** Define stopping criteria, such as reaching a specific region size, exploring all neighbors, or exceeding a certain intensity difference threshold. These criteria determine when the region growing process stops.

**Applications:**

- **Medical Imaging:** Region Growing is used in medical image analysis for segmenting organs and lesions from scans such as MRI and CT.
- **Remote Sensing:** It's applied in satellite image analysis for land cover classification and feature extraction.
- **Computer Vision:** Region Growing is used in object detection and tracking, where homogeneous regions represent distinct objects.
- **Image Editing:** Applied in applications like image inpainting and image editing for selecting and modifying specific regions.

**Advantages:**

- **Homogeneity:** Region Growing produces homogeneous regions based on pixel similarity criteria, making it effective for segmenting objects with uniform properties.
- **Flexibility:** It can be adapted for various types of images and segmentation tasks by adjusting the similarity criterion and stopping criteria.

**Considerations:**

- **Seed Selection:** The choice of seed points significantly affects the segmentation result. Proper seed selection is crucial for accurate segmentation.
- **Computational Complexity:** Depending on the size of the image and the complexity of the criteria, region growing can be computationally intensive.

Region Growing is a versatile and intuitive segmentation technique, capable of producing coherent regions from images based on local pixel properties, making it valuable for a wide range of image analysis applications.
"""

Region_Splitting_Text = """
**Region Splitting Method for Image Segmentation:**

Region Splitting is a recursive image segmentation technique used in image processing to partition an image into homogenous regions. The method divides the image into smaller regions and recursively checks the homogeneity of these regions based on certain criteria. If a region is not homogeneous, it is further split into sub-regions until a stopping criterion is met.

**Algorithm:**

1. **Initial Split:** Start by considering the entire image as a single region.
   
2. **Homogeneity Check:** Examine the homogeneity of the current region using specific criteria, such as intensity variation, texture, or color differences. If the region is considered homogeneous according to the criteria, no further splitting is performed.

3. **Region Splitting:** If the region is not homogeneous, split it into sub-regions. Common splitting methods include dividing the region into quadrants or halves. Repeat the homogeneity check for each sub-region.

4. **Recursive Splitting:** For each non-homogeneous sub-region, repeat the splitting process recursively until the stopping criterion is met. This criterion could be a predefined threshold for homogeneity or a minimum region size.

5. **Stopping Criterion:** The algorithm stops when the regions meet the homogeneity criteria, or when they are smaller than a specified size, ensuring that the segmentation process does not continue indefinitely.

**Advantages:**

- **Adaptability:** Region Splitting can adapt to varying levels of homogeneity, making it suitable for images with complex structures and textures.
- **Hierarchical Segmentation:** The recursive nature allows the generation of hierarchical segmentations, providing detailed information about different scales of structures in the image.

**Considerations/Challenges:**

- **Computational Complexity:** The recursive nature of Region Splitting can make it computationally intensive, especially for large images, and requires careful optimization.
- **Initial Parameters:** Proper selection of initial parameters, such as splitting criteria and stopping conditions, is crucial for obtaining meaningful segmentation results.

Region Splitting is a versatile segmentation method that provides hierarchical and adaptive segmentation results, making it valuable in applications where detailed information about different regions of an image is needed. Careful consideration of parameters and stopping conditions is essential for its successful implementation.
"""

st.set_page_config(
    page_title="About - Advanced Image Processor",
    page_icon="🖼️",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.sidebar.title("💡About")
st.sidebar.subheader("This is a web app to perform and learn about image processing techniques on a given input image.")
if st.sidebar.button("ℹ️ Home Page"):
   switch_page("Advanced_Image_Processor")

st.sidebar.title("👨🏽‍💻Developer")
st.sidebar.info(
    "This app is created by **Aadith Sukumar**\n"
    "\n🤝🏽[**Connect on LinkedIn**](https://www.linkedin.com/in/aadith-sukumar) \n"
    "\n👾[**Follow on GitHub**](https://www.github.com/aadi1011)\n"
)

st.title("Advanced Image Processor")

st.markdown("---")

method_choice = st.selectbox("Choose an Image Processing Method to Learn More About:",("About The Project", 
                                                                                          "Fourier Domain Transform", 
                                                                                          "Edge Detection",
                                                                                          "Image Segmentation",
                                                                                          "Region-Based Methods"))

if method_choice == "About The Project":
   st.header("About the Project")
   st.markdown(About_Project_Text)
   st.markdown(
    "[![X (formerly Twitter) Follow](https://img.shields.io/twitter/follow/aadi0x01)](https://twitter.com/aadi0x01)"
    "&nbsp;&nbsp; [![GitHub followers](https://img.shields.io/github/followers/aadi1011?label=Followers&logo=github&color=white)](https://github.com/aadi1011) "
    "&nbsp;&nbsp; [![Last Updated](https://img.shields.io/github/last-commit/aadi1011/Advanced-Image-Processor/main?label=Last%20Updated)](https://github.com/aadi1011/Advanced-Image-Processor)"
   )





elif method_choice == "Fourier Domain Transform":
   st.header("Fourier Domain Transform")
   technique_choice = st.selectbox("Choose a Fourier Domain Transform Technique to Learn More About:",("Fourier Domain Transform",
                                                                                                         "Butterworth Filters", 
                                                                                                         "Blurring Technique",
                                                                                                         "Sharpening Technique",
                                                                                                         "Gaussian Noise",
                                                                                                         "Salt and Pepper Noise"))
   
   if technique_choice == "Fourier Domain Transform":
      st.markdown("---")
      st.markdown(Fourier_Domain_Text)
      st.markdown("---")

   elif technique_choice == "Butterworth Filters":
      st.markdown("---")
      st.markdown(Butterworth_Filter_Text)
      st.markdown("---")

   elif technique_choice == "Blurring Technique":
      st.markdown("---")
      st.markdown(Blur_Text)
      st.markdown("---")

   elif technique_choice == "Sharpening Technique":
      st.markdown("---")
      st.markdown(Sharpen_Text)
      st.markdown("---")

   elif technique_choice == "Gaussian Noise":
      st.markdown("---")
      st.markdown(Gaussian_Noise_Text)
      st.markdown("---")

   elif technique_choice == "Salt and Pepper Noise":
      st.markdown("---")
      st.markdown(Salt_Pepper_Noise_Text)
      st.markdown("---")

elif method_choice == "Edge Detection":
   st.header("Edge Detection")
   technique_choice = st.selectbox("Choose an Edge Detection Technique to Learn More About:",("Edge Detection",
                                                                                               "Sobel Operator", 
                                                                                               "Prewitt Operator",
                                                                                               "Roberts Cross Operator",
                                                                                               "Canny Edge Detector"))
   
   if technique_choice == "Edge Detection":
      st.markdown("---")
      st.markdown(Edge_Detection_Text)
      st.markdown("---")
   
   elif technique_choice == "Sobel Operator":
      st.markdown("---")
      st.markdown(Sobel_Text)
      st.markdown("---")

   elif technique_choice == "Prewitt Operator":
      st.markdown("---")
      st.markdown(Prewitt_Text)
      st.markdown("---")

   elif technique_choice == "Roberts Cross Operator":
      st.markdown("---")
      st.markdown(Roberts_Text)
      st.markdown("---")

   elif technique_choice == "Canny Edge Detector":
      st.markdown("---")
      st.markdown(Canny_Text)
      st.markdown("---")

elif method_choice == "Image Segmentation":
   st.header("Image Segmentation")
   technique_choice = st.selectbox("Choose an Image Segmentation Technique to Learn More About:",("Peak Signal-to-Noise Ratio",
                                                                                                    "Binary Thresholding",
                                                                                                    "Truncated Thresholding",
                                                                                                    "To Zero Thresholding",
                                                                                                    "Otsu's Thresholding",
                                                                                                    "Gaussian Thresholding",
                                                                                                    "Adaptive Thresholding",
                                                                                                    "Region Growing Method",
                                                                                                    "Region Splitting Method"))
   
   if technique_choice == "Peak Signal-to-Noise Ratio":
      st.markdown("---")
      st.markdown(PSNR_Text)
      st.markdown("---")

   elif technique_choice == "Binary Thresholding":
      st.markdown("---")
      st.markdown(Binary_Thresholding_Text)
      st.markdown("---")

   elif technique_choice == "Truncated Thresholding":
      st.markdown("---")
      st.markdown(Truncated_Thresholding_Text)
      st.markdown("---")

   elif technique_choice == "To Zero Thresholding":
      st.markdown("---")
      st.markdown(To_Zero_Thresholding_Text)
      st.markdown("---")

   elif technique_choice == "Otsu's Thresholding":
      st.markdown("---")
      st.markdown(Otsu_Thresholding_Text)
      st.markdown("---")

   elif technique_choice == "Gaussian Thresholding":
      st.markdown("---")
      st.markdown(Gaussian_Thresholding_Text)
      st.markdown("---")

   elif technique_choice == "Adaptive Thresholding":
      st.markdown("---")
      st.markdown(Adaptive_Thresholding_Text)
      st.markdown("---")

   elif technique_choice == "Region Growing Method":
      st.markdown("---")
      st.markdown(Region_Growing_Text)
      st.markdown("---")

   elif technique_choice == "Region Splitting Method":
      st.markdown("---")
      st.markdown(Region_Splitting_Text)
      st.markdown("---")

elif method_choice == "Region-Based Methods":
   st.markdown("---")
   st.header("Region-Based Methods")
   st.markdown("---")
   st.markdown(Region_Growing_Text)
   st.markdown("---")
   st.markdown(Region_Splitting_Text)
   st.markdown("---")
