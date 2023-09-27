import numpy as np
import matplotlib
import time
from helpers import progressbar
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.transform import resize
from scipy.spatial.distance import cdist
import scipy
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC

'''
READ FIRST: Relationship Between Functions

Functions in this file can be classified into 3 groups based on their roles:
Group 1: feature-extracting functions
        a) get_tiny_images: 
            read in the images from the input paths and size down the images; 
            the output tiny images are used as features
        b) get_bags_of_words: 
            read in the images from the input paths and 
            turn each of the images into a histogram of oriented gradients (hog); 
            the output histograms are used as features
Group 2: supplementary function for get_bags_of_words (the second function in Group 1)
        build_vocabulary:
            read in the images from the input paths and build a vocabulary using the images using K-Means;
            the output vocabulary are fed into get_bags_of_words
            (Only need to run this function in main.py once)
Group 3: classification functions
        a) nearest_neighbor_classify
            implement nearest-neighbor classifier
        b) svm_classify
            implement many-versus-one linear SVM classifier

In main.py, we will run different combinations of functions in Group 1 and Group 3, e.g.
    i) get_tiny_images + nearest_neighbor_classify    
    ii) get_bags_of_words + nearest_neighbor_classify
    iii) get_bags_of_words + svm_classify
    to perform scene classification.
We recommend to implement the functions in the following order:
    1) get_tiny_images, nearest_neighbor_classify, THEN run (i) to see the performance;
    2) get_bags_of_words, THEN run (ii) to see the performance.
    3) svm_classify, THEN run (iii) to see the performance.

Read main.py for more details.
'''

def get_tiny_images(image_paths):
    '''
    This feature is inspired by the simple tiny images used as features in
    80 million tiny images: a large dataset for non-parametric object and
    scene recognition. A. Torralba, R. Fergus, W. T. Freeman. IEEE
    Transactions on Pattern Analysis and Machine Intelligence, vol.30(11),
    pp. 1958-1970, 2008. http://groups.csail.mit.edu/vision/TinyImages/

    Inputs:
        image_paths: a 1-D Python list of strings. Each string is a complete
                     path to an image on the filesystem.
    Outputs:
        An n x d numpy array where n is the number of images and d is the
        length of the tiny image representation vector. e.g. if the images
        are resized to 16x16, then d is 16 * 16 = 256.

    To build a tiny image feature, resize the original image to a very small
    square resolution (e.g. 16x16). You can either resize the images to square
    while ignoring their aspect ratio, or you can crop the images into squares
    first and then resize evenly. Normalizing these tiny images will increase
    performance modestly.

    As you may recall from class, naively downsizing an image can cause
    aliasing artifacts that may throw off your comparisons. See the docs for
    skimage.transform.resize for details:
    http://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.resize

    Suggested functions: skimage.transform.resize, skimage.color.rgb2grey,
                         skimage.io.imread, np.reshape
    '''

    tiny_images = []
    for i in image_paths:
        image = imread(i, as_gray = True)
        tiny_image = resize(image, (16,16), anti_aliasing = True)
        tiny_images.append(tiny_image)
    
    tiny_images = np.array(tiny_images)
    tiny_images = np.reshape(tiny_images,(-1,16*16))
    # print(tiny_images.shape)
    return tiny_images

def build_vocabulary(image_paths, vocab_size):
    '''
    This function samples HOG descriptors from the training images,
    cluster them with kmeans, and then return the cluster centers.

    Inputs:
        image_paths: a Python list of image path strings
        vocab_size: an integer indicating the number of words desired for the
                     bag of words vocab set

    Outputs:
        a vocab_size x (z*z*9) (see below) array which contains the cluster
        centers that result from the K Means clustering.

    You'll need to generate HOG features using the skimage.feature.hog() function.
    The documentation is available here:
    http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.hog

    We will highlight some
    important arguments to consider:
        cells_per_block: The hog function breaks the image into evenly-sized
            blocks, which are further broken down into cells, each made of
            pixels_per_cell pixels (see below). Setting this parameter tells the
            function how many cells to include in each block. This is a tuple of
            width and height. Your SIFT implementation, which had a total of
            16 cells, was equivalent to setting this argument to (4,4).
        pixels_per_cell: This controls the width and height of each cell
            (in pixels). Like cells_per_block, it is a tuple. In your SIFT
            implementation, each cell was 4 pixels by 4 pixels, so (4,4).
        feature_vector: This argument is a boolean which tells the function
            what shape it should use for the return array. When set to True,
            it returns one long array. We recommend setting it to True and
            reshaping the result rather than working with the default value,
            as it is very confusing.

    It is up to you to choose your cells per block and pixels per cell. Choose
    values that generate reasonably-sized feature vectors and produce good
    classification results. For each cell, HOG produces a histogram (feature
    vector) of length 9. We want one feature vector per block. To do this we
    can append the histograms for each cell together. Let's say you set
    cells_per_block = (z,z). This means that the length of your feature vector
    for the block will be z*z*9.

    With feature_vector=True, hog() will return one long np array containing every
    cell histogram concatenated end to end. We want to break this up into a
    list of (z*z*9) block feature vectors. We can do this using a numpy
    function. When using np.reshape, you can set the length of one dimension to
    -1, which tells numpy to make this dimension as big as it needs to be to
    accomodate to reshape all of the data based on the other dimensions. So if
    we want to break our long np array (long_boi) into rows of z*z*9 feature
    vectors we can use small_bois = long_boi.reshape(-1, z*z*9).


    ONE MORE THING
    If we returned all the features we found as our vocabulary, we would have an
    absolutely massive vocabulary. That would make matching inefficient AND
    inaccurate! So we use K Means clustering to find a much smaller (vocab_size)
    number of representative points. We recommend using sklearn.cluster.KMeans
    or sklearn.cluster.MiniBatchKMeans if sklearn.cluster.KMeans takes to long for you. 
    Note that this can take A LONG TIME to complete (upwards of ten minutes 
    for large numbers of features and large max_iter), so set the max_iter argument
    to something low (we used 100) and be patient. You may also find success setting
    the "tol" argument (see documentation for details).

    Once the vocabulary is created, it is saved as vocab.npy in a call in main.py. 
    If you then use the flag `--load_vocab` on launch, it will load the vocab instead
    of recreating it. Hey presto!
    '''

    
    num_imgs = len(image_paths)
    hog_all = []

    for i in progressbar(range(num_imgs), "Loading ...", num_imgs):
        image = imread(image_paths[i], as_gray = True)
        hog_features = hog(image, orientations=9, cells_per_block=(4,4), pixels_per_cell=(4,4), feature_vector=True)  # (535824,); one big hog feature 
        hog_features = np.reshape(hog_features, (-1,4*4*9))  # (3721, 144)
        # print(hog_features.shape)
        hog_all.append(hog_features)
    hog_all = np.array(hog_all) # (1500,); 1500 images in total 
    hog_all = np.vstack(hog_all) # (5639973, 144); there are total 5639973 feature points 
    # np.vstack: Stack arrays in sequence vertically (row wise).s
    # print("this is hog_all.shape:")
    # print(hog_all.shape)

    kmeans_vocab = KMeans(n_clusters = vocab_size, max_iter = 10).fit(hog_all)
    print("this is vocab size:")
    print(vocab_size) # 200; number of centers 
    vocab_center = kmeans_vocab.cluster_centers_
    vocab_center = np.array(vocab_center)
    print("this is vocab_center.shape:")
    print(vocab_center.shape) # (200, 144)
    return vocab_center 

def get_bags_of_words(image_paths):
    '''
    This function should take in a list of image paths and calculate a bag of
    words histogram for each image, then return those histograms in an array.

    Inputs:
        image_paths: A Python list of strings, where each string is a complete
                     path to one image on the disk.

    Outputs:
        An nxd numpy matrix, where n is the number of images in image_paths and
        d is size of the histogram built for each image.

    Use the same hog function to extract feature vectors as before (see
    build_vocabulary). It is important that you use the same hog settings for
    both build_vocabulary and get_bags_of_words! Otherwise, you will end up
    with different feature representations between your vocab and your test
    images, and you won't be able to match anything at all!

    After getting the feature vectors for an image, you will build up a
    histogram that represents what words are contained within the image.
    For each feature, find the closest vocab word, then add 1 to the histogram
    at the index of that word. For example, if the closest vector in the vocab
    is the 103rd word, then you should add 1 to the 103rd histogram bin. Your
    histogram should have as many bins as there are vocabulary words.

    Suggested functions: scipy.spatial.distance.cdist, np.argsort,
                         np.linalg.norm, skimage.feature.hog
    '''

    vocab = np.load('vocab.npy') # (200, 144)
    print('Loaded vocab from file.')



    hist = np.zeros((np.shape(image_paths)[0], np.shape(vocab)[0]))
    # print(hist.shape) # (1500, 200)
    for i in range(np.shape(image_paths)[0]):
        image = imread(image_paths[i])
        hog_features = hog(image, orientations=9, cells_per_block=(4,4), pixels_per_cell=(4,4), feature_vector=True)
        hog_features = np.reshape(hog_features, (-1,4*4*9))
        distance = cdist(hog_features, vocab, metric='euclidean')
        # print(distance.shape) # (3640, 200)

        for j in range(np.shape(distance)[0]):
            closest = np.argsort(distance[j])[0]
            # print("this is closest:")
            # print(closest.shape) # (200,)
            hist[i][closest] += 1
    
    hist = normalize(hist)
    hist = np.array(hist) # (1500, 200)
    
    return hist 


def svm_classify(train_image_feats, train_labels, test_image_feats):
    '''
    This function will predict a category for every test image by training
    15 many-versus-one linear SVM classifiers on the training data, then
    using those learned classifiers on the testing data.

    Inputs:
        train_image_feats:  An nxd numpy array, where n is the number of training
                            examples, and d is the image descriptor vector size.
        train_labels:       An n x 1 Python list containing the corresponding ground
                            truth labels for the training data.
        test_image_feats:   An m x d numpy array, where m is the number of test
                            images and d is the image descriptor vector size.

    Outputs:
        An m x 1 numpy array of strings, where each string is the predicted label
        for the corresponding image in test_image_feats

    EXTRA CREDIT up to +5
    Implement a 15-way classifier using multiple binary (2-way) classifier functions, 
    as we did in pseudocode in the written questions.
    '''


    svm_result = LinearSVC(random_state=0, tol=1e-5).fit(train_image_feats, train_labels)
    predictions = svm_result.predict(test_image_feats)
    return predictions

def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats):
    '''
    This function will predict the category for every test image by finding
    the training image with most similar features. You will complete the given
    partial implementation of k-nearest-neighbors such that for any arbitrary
    k, your algorithm finds the closest k neighbors and then votes among them
    to find the most common category and returns that as its prediction.

    Inputs:
        train_image_feats: An nxd numpy array, where n is the number of training
                           examples, and d is the image descriptor vector size.
        train_labels: An nx1 Python list containing the corresponding ground
                      truth labels for the training data.
        test_image_feats: An mxd numpy array, where m is the number of test
                          images and d is the image descriptor vector size.

    Outputs:
        An mx1 numpy list of strings, where each string is the predicted label
        for the corresponding image in test_image_feats

    The simplest implementation of k-nearest-neighbors gives an even vote to
    all k neighbors found - each neighbor in category A counts as one
    vote for category A, and the result returned is equivalent to finding the
    mode of the categories of the k nearest neighbors. A more advanced version
    uses weighted votes where closer matches matter more strongly than far ones.
    This is not required, but may increase performance.

    Be aware that increasing k does not always improve performance. Play around
    with a few values and see what changes.

    Useful functions:
        scipy.spatial.distance.cdist, np.argsort, scipy.stats.mode
    '''

    k = 10
    final_predictions = []

    # Gets the distance between each test image feature and each train image feature
    # e.g., cdist
    distances = cdist(test_image_feats, train_image_feats, 'euclidean')
    print(distances.shape) # (1500, 1500)

    for i in range(test_image_feats.shape[0]):
        best = distances[i].argsort()[:k]
        best_labels = np.take(train_labels, best)
        mode = scipy.stats.mode(best_labels)[0]
        final_predictions.append(mode)

    final_predictions = np.array(final_predictions)


    return final_predictions
