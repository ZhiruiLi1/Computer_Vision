import numpy as np
from numpy import pi, exp, sqrt
from skimage import io, img_as_ubyte, img_as_float32
from skimage.transform import rescale

def my_imfilter(image, kernel):
    """
    Your function should meet the requirements laid out on the homework webpage.
    Apply a filter (using kernel) to an image. Return the filtered image. To
    achieve acceptable runtimes, you MUST use numpy multiplication and summation
    when applying the kernel.
    Inputs
    - image: numpy nd-array of dim (m,n) or (m, n, c)
    - kernel: numpy nd-array of dim (k, l)
    Returns
    - filtered_image: numpy nd-array of dim of equal 2D size (m,n) or 3D size (m, n, c)
    Errors if:
    - filter/kernel has any even dimension -> raise an Exception with a suitable error message.
    """
    filtered_image = np.zeros(image.shape)


    # Initialization:
    
    assert kernel.shape[0] % 2 == 1, "The first dimension of the kernel must be odd!"
    assert kernel.shape[1] % 2 == 1, "The second dimension of the kernel must be odd!"
    
    m = image.shape[0]
    n = image.shape[1]
    if len(image.shape) == 3:
        c = image.shape[2]
    else:
        c = None
    k = kernel.shape[0]  # height
    l = kernel.shape[1]  # width 

    # Padding:
    row_pad = (l-1)//2
    col_pad = (k-1)//2

    if c is not None:
        final_image = np.pad(image, [(col_pad, col_pad), (row_pad, row_pad), (0,0)], mode="reflect") 
        # mode = "constant": pad with 0s
        # first dimension, second dimension, and third dimension 
    else:
        final_image = np.pad(image, [(col_pad, col_pad), (row_pad, row_pad)], mode="reflect")

    # Rotate the kernel for convolution:
    rotated_k = np.rot90(kernel, k = 2)
    # k: number of times the array is rotated by 90 degrees

    # Convolution:
    filtered_image = np.zeros(image.shape)
    if c is not None:
        for cha in range(c):
            for row in range(m):
                for col in range (n):
                    filtered_image[row][col][cha] = np.sum(final_image[row:row+k, col:col+l, cha] * rotated_k)
    else:
        for row in range(m):
                for col in range (n):
                    filtered_image[row][col] = np.sum(final_image[row:row+k, col:col+l] * rotated_k)


				
                    

    ##################

    return filtered_image

"""
EXTRA CREDIT placeholder function
"""

def my_imfilter_fft(image, kernel):
    """
    Your function should meet the requirements laid out in the extra credit section on
    the homework webpage. Apply a filter (using kernel) to an image. Return the filtered image.
    Inputs
    - image: numpy nd-array of dim (m,n) or (m, n, c)
    - kernel: numpy nd-array of dim (k, l)
    Returns
    - filtered_image: numpy nd-array of dim of equal 2D size (m,n) or 3D size (m, n, c)
    Errors if:
    - filter/kernel has any even dimension -> raise an Exception with a suitable error message.
    """
    filtered_image = np.zeros(image.shape)

    ##################
    print('my_imfilter_fft function in student.py is not implemented')
    ##################

    return filtered_image


def gen_hybrid_image(image1, image2, cutoff_frequency):
    """
     Inputs:
     - image1 -> The image from which to take the low frequencies.
     - image2 -> The image from which to take the high frequencies.
     - cutoff_frequency -> The standard deviation, in pixels, of the Gaussian
                           blur that will remove high frequencies.

     Task:
     - Use my_imfilter to create 'low_frequencies' and 'high_frequencies'.
     - Combine them to create 'hybrid_image'.
    """
   

    assert image1.shape[0] == image2.shape[0]
    assert image1.shape[1] == image2.shape[1]
    assert image1.shape[2] == image2.shape[2]

    # Steps:
    # (1) Remove the high frequencies from image1 by blurring it. The amount of
    #     blur that works best will vary with different image pairs
    # generate a 1x(2k+1) gaussian kernel with mean=0 and sigma = s, see https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python
    s, k = cutoff_frequency, cutoff_frequency*2
    probs = np.asarray([exp(-z*z/(2*s*s))/sqrt(2*pi*s*s) for z in range(-k,k+1)], dtype=np.float32)
    kernel = np.outer(probs, probs)

    low_frequencies = my_imfilter(image1, kernel)

    # (2) Remove the low frequencies from image2. The easiest way to do this is to
    #     subtract a blurred version of image2 from the original version of image2.
    #     This will give you an image centered at zero with negative values.
    high_frequencies = image2 = my_imfilter(image2, kernel) 
    
    # (3) Combine the high frequencies and low frequencies
    hybrid_image = low_frequencies + high_frequencies
    hybrid_image = np.clip(hybrid_image, 0, 1)
    return low_frequencies, high_frequencies, hybrid_image
