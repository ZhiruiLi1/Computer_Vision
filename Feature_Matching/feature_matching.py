from re import X
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, feature, img_as_int
from skimage.measure import regionprops
from sklearn.preprocessing import normalize

def plot_interest_points(image, x, y):
    '''
    Plot interest points for the input image. 
    
    Show the interest points given on the input image. Be sure to add the images you make to your writeup. 

    Useful functions: Some helpful (not necessarily required) functions may include
        - matplotlib.pyplot.imshow, matplotlib.pyplot.scatter, matplotlib.pyplot.show, matplotlib.pyplot.savefig
    
    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :x: np array of x coordinates of interest points
    :y: np array of y coordinates of interest points
    '''

    # TODO: Your implementation here! See block comments and the homework webpage for instructions
    plt.imshow(image)
    plt.plot(x,y)
    plt.show()

def get_feature_points(image, feature_width):
    '''
    Returns interest points for the input image

    (Please note that we recommend implementing this function last and using cheat_interest_points()
    to test your implementation of get_feature_descriptors() and match_features())

    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    You do not need to worry about scale invariance or keypoint orientation estimation
    for your Harris corner detector.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.

    If you're finding spurious (false/fake) interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.feature.peak_local_max (experiment with different min_distance values to get good results)
        - skimage.measure.regionprops
          
    Note: You may decide it is unnecessary to use feature_width in get_feature_points, or you may also decide to 
    use this parameter to exclude the points near image edges.

    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :feature_width: the width and height of each local feature in pixels

    :returns:
    :xs: an np array of the x coordinates of the interest points in the image
    :ys: an np array of the y coordinates of the interest points in the image

    :optional returns (may be useful for extra credit portions):
    :confidences: an np array indicating the confidence (strength) of each interest point
    :scale: an np array indicating the scale of each interest point
    :orientation: an np array indicating the orientation of each interest point

    '''

    # TODO: Your implementation here! See block comments and the homework webpage for instructions

    # These are placeholders - replace with the coordinates of your interest points!

    xs = np.zeros(1)
    ys = np.zeros(1)
    # Note that xs and ys represent the coordinates of the image. Thus, xs actually denote the columns
    # of the respective points and ys denote the rows of the respective points.

    # STEP 1: Calculate the gradient (partial derivatives on two directions).
    grad_x = filters.sobel_v(image) # vertical sobel filter # calculate horizontal gradient (along x direction)
    grad_y = filters.sobel_h(image) # horizontal sobel filter # calculate vertical gradient (along y direction)

    grad_x2 = grad_x * grad_x # second order derivative along x direction
    grad_y2 = grad_y * grad_y # second order derivative along y direction 
    grad_xy = grad_x * grad_y # second order derivative along xy direction 
    # the convolution of a 1st order Sobel filter kernel with another 1st order Sobel filter gives the filter kernel for a second order filter

    # STEP 2: Apply Gaussian filter with appropriate sigma.
    grad_x2_g = filters.gaussian(grad_x2, sigma = 1)
    grad_y2_g = filters.gaussian(grad_y2, sigma = 1)
    grad_xy_g = filters.gaussian(grad_xy, sigma = 1)

    # STEP 3: Calculate Harris cornerness score for all pixels.
    alpha = 0.1
    C = grad_x2_g * grad_y2_g - grad_xy_g**2 - alpha*((grad_x2_g + grad_y2_g)**2)

    # STEP 4: Peak local max to eliminate clusters. (Try different parameters.)
    xs = feature.peak_local_max(C, min_distance = 10, threshold_rel=0.01)[:,1] # columns
    ys = feature.peak_local_max(C, min_distance = 10, threshold_rel=0.01)[:,0] # rows 
    # BONUS: There are some ways to improve:
    # 1. Making interest point detection multi-scaled.
    # 2. Use adaptive non-maximum suppression.

    return xs, ys


def get_feature_descriptors(image, x_array, y_array, feature_width):
    '''
    Returns features for a given set of interest points.

    To start with, you might want to simply use normalized patches as your
    local feature descriptor. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT-like feature descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) feature descriptor should have:
    (1) a 4x4 grid of cells, each feature_width / 4 pixels square.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like features can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.filters (library)


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :x: np array of x coordinates of interest points
    :y: np array of y coordinates of interest points
    :feature_width: in pixels, is the local feature width. You can assume
                    that feature_width will be a multiple of 4 (i.e. every cell of your
                    local SIFT-like feature will have an integer width and height).
    If you want to detect and describe features at multiple scales or
    particular orientations you can add input arguments. Make sure input arguments 
    are optional or the autograder will break.

    :returns:
    :features: numpy array of computed features. It should be of size
            [num points * feature dimensionality]. For standard SIFT, `feature
            dimensionality` is 128. `num points` may be less than len(x) if
            some points are rejected, e.g., if out of bounds.

    '''

    # TODO: Your implementation here! See block comments and the homework webpage for instructions

    # This is a placeholder - replace this with your features!
    features = np.zeros((len(x_array),128))

    # STEP 1: Calculate the gradient (partial derivatives on two directions) on all pixels.
    grad_x = filters.sobel_v(image) # vertical sobel filter # calculate horizontal gradient (along x direction)
    grad_y = filters.sobel_h(image) # horizontal sobel filter # calculate vertical gradient (along y direction)


    # STEP 2: Decompose the gradient vectors to magnitude and direction.
    mag = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    ori = np.arctan2(grad_y, grad_x) # element-wise arc tangent of x1/x2 choosing the quadrant correctly
    print(f'this is ori: {ori}')
    print(f'this is mag: {mag}')
    # STEP 3: For each interest point, calculate the local histogram based on related 4x4 grid cells.
    #         Each cell is a square with feature_width / 4 pixels length of side.
    #         For each cell, we assign these gradient vectors corresponding to these pixels to 8 bins
    #         based on the direction (angle) of the gradient vectors. 
    # STEP 4: Now for each cell, we have a 8-dimensional vector. Appending the vectors in the 4x4 cells,
    #         we have a 128-dimensional feature.
    # STEP 5: Don't forget to normalize your feature.
    
    # feature_width = 16 
    width = feature_width // 4 # 4 
    half_width = feature_width // 2 # 8
    
    ori = 4*(ori+np.max(ori)) // np.pi # convert orientation to bins 
    ori[ori==8] = 7 # optional 

    print(f"this is the list of unique values: {np.unique(ori)}")
    print(f'this is ori: {ori}')
    print(f'this is ori[3][4]: {ori[3][4]}')
    
    # x_array records columns
    # y_array records rows 
    for i in range(len(x_array)):
        count = 0
        for j in range(x_array[i] - half_width, x_array[i] + half_width, width):  # (-8, -4, 0 ,4)
            for k in range(y_array[i] - half_width, y_array[i] + half_width, width):  # (-8, -4, 0 ,4)
                des = np.zeros(8)
                for u in range(j, j+width): # (-8, -4)
                    for v in range(k, k+width):
                        des[int(ori[v][u])] = des[int(ori[v][u])] + mag[v][u]  # sum up all the magnitudes for each direction (total 8)

                features[i, 8*count:8*(count+1)] = des
                count += 1
    features = normalize(features)




    # BONUS: There are some ways to improve:
    # 1. Use a multi-scaled feature descriptor.
    # 2. Borrow ideas from GLOH or other type of feature descriptors.

    print(f"this is features[0]: {features[0]}")
    print(f"this is the shape of features: {features.shape}") # (428, 128)
    return features 


def match_features(im1_features, im2_features):
    '''
    Implements the Nearest Neighbor Distance Ratio Test to assign matches between interest points
    in two images.

    Please implement the "Nearest Neighbor Distance Ratio (NNDR) Test" ,
    Equation 4.18 in Section 4.1.3 of Szeliski.

    For extra credit you can implement spatial verification of matches.

    Please assign a confidence, else the evaluation function will not work. Remember that
    the NNDR test will return a number close to 1 for feature points with similar distances.
    Think about how confidence relates to NNDR.

    This function does not need to be symmetric (e.g., it can produce
    different numbers of matches depending on the order of the arguments).

    A match is between a feature in im1_features and a feature in im2_features. We can
    represent this match as a the index of the feature in im1_features and the index
    of the feature in im2_features

    :params:
    :im1_features: an np array of features returned from get_feature_descriptors() for interest points in image1
    :im2_features: an np array of features returned from get_feature_descriptors() for interest points in image2

    :returns:
    :matches: an np array of dimension k x 2 where k is the number of matches. The first
            column is an index into im1_features and the second column is an index into im2_features
    :confidences: an np array with a real valued confidence for each match
    '''

    # TODO: Your implementation here! See block comments and the project webpage for instructions

    # These are placeholders - replace with your matches and confidences!
    
    # STEP 1: Calculate the distances between each pairs of features between im1_features and im2_features.
    #         HINT: https://browncsci1430.github.io/webpage/hw2_featurematching/efficient_sift/

    print(f'this is the shape of im1_features: {im1_features.shape}') # (366, 128)
    print(f'this is the shape of im2_features: {im2_features.shape}') # (428, 128)
    im1_square = im1_features * im1_features
    im2_square = im2_features * im2_features

    im1_square_sum = np.sum(im1_square, axis = 1)
    im1_square_sum = np.expand_dims(im1_square_sum, axis = 1)
    im2_square_sum = np.sum(im2_square, axis = 1)
    im2_square_sum = np.expand_dims(im2_square_sum, axis = 0)

    A = im1_square_sum + im2_square_sum
    B = 2 * np.matmul(im1_features, np.transpose(im2_features))
    D = np.sqrt(A - B) #this is the shape of D: (366, 428)
    print(f'this is the shape of D: {D.shape}')
    print(D)
    # STEP 2: Sort and find closest features for each feature, then performs NNDR test.
    
    sorted_indices = np.argsort(D, axis = -1) # returns the indices that would sort an array
    # entry (i,j) of the distance matrix should represent the distance between feature i of the first image and feature j of the second
    sorted_distances = np.take_along_axis(D, sorted_indices, axis = -1) 


    nearest_i = sorted_indices[:, 0]
    nearest_d = sorted_distances[:, 0]
    second_d = sorted_distances[:, 1]


    confidences = -(nearest_d / (second_d + 0.0001))

    matches = np.stack([np.array(range(D.shape[0])), nearest_i], axis=0)
    matches = np.transpose(matches)
    print(f'this is the shape of matches: {matches.shape}') # (366, 2)
    # BONUS: Using PCA might help the speed (but maybe not the accuracy).
    return matches, confidences
