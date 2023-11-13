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

\- [**Aadith Sukumar**](www.github.com/aadi1011)
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
