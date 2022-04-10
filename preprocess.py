import cv2
import numpy as np
import matplotlib.pyplot as plt

def resize_images(dataset, size = 800, debug = 0):
    '''
    Proportionally resizes a dataset of images to the specified height.

    Parameters:
        dataset (array): An array of np.array images - the dataset of images to resize
        size (int): The height to resize the image to (default = 800)
        debug (int): Boolean value, whether or not to print sample images to visualize the process (default = 0)

    Returns:
        resized (np.array): The resized image
    '''
    
    # Resize image (standardize)
    max_length = size
    resized_dataset = []
    
    for img in dataset:
        ratio = max_length / img.shape[0]
        dim = (int(img.shape[1] * ratio), max_length)

        # resize image
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        resized_dataset.append(resized)

        if debug:
            print('Resized Dimensions : ',resized.shape)

            plt.imshow(resized, cmap = plt.get_cmap('gray'))
            plt.show()

    return resized_dataset



def binarize_images(dataset, debug = 0):
    '''
    Creates an inverse binary dataset of images (0s for whitespace, 1s for script image data).
    Returns BOTH binary and inverse binary images as a tuple.

    Parameters:
        dataset (array): An array of np.array images - the dataset of image files (grayscale) to process
        debug (int): Boolean value, whether or not to print sample images to visualize the process (default = 0)

    Returns:
        bin (np.array): The binarized (from grayscale) image (0s for script, 1s for whitespace)
        ibin (np.array): The inverse binary image (0s for whitespace, 1s for script)
    '''
    
    bin_dataset = []
    ibin_dataset = []
    
    for img in dataset:
        # Create binary image (only 1s and 0s) using threshold
        ret, bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bin_dataset.append(bin)

        # inverse binary image (black bg, white txt)
        ibin = cv2.bitwise_not(bin)
        ibin_dataset.append(ibin)

        if debug:
            # Visualize both images.
            f, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 20))
            ax1.set_title('Binarized Image')
            ax1.imshow(bin, cmap = plt.get_cmap('gray'))
            ax2.set_title('Inversed Binary Image')
            ax2.imshow(ibin, cmap = plt.get_cmap('gray'))
            plt.show()
    
    return bin_dataset, ibin_dataset



def count_nonzero_values(img, debug = 0):
    '''
    Locates font areas in a script image.

    Parameters:
        img (array): An np.array image
        debug (int): Boolean value, whether or not to print sample images to visualize the process (default = 0)

    Returns:
        n_fontarea (int): The number of font areas (lines of script, or individual words) that were found in the image
        cols (array): An array with the non-zero values determined for each column in the image
    '''
    
    # Var to hold num of font areas (vertical lines of text) based on the script
    n_fontarea = 0
    height, width = img.shape
    ibin = cv2.bitwise_not(img)
    
    # Create an array of 0s to represent the entire width of the image (459px or etc)
    cols = np.full(width, 0)
    
    if debug:
        print('Image width = ', len(cols))
        print('\nBefore scanning:\n', cols)

    # For every col in the array, find anything with data (a non-zero value, i.e. a pixel of the script)
    for i in range(width):
        cols[i] = cv2.countNonZero(ibin[:, i])
    
    if debug:
        print('\nAfter scanning:\n', cols)

    # Determine font areas by checking where a non-zero col ends and a zero col begins
    for i in range(width - 1):
        if cols[i] > 0 and cols[i+1] == 0: # here, our script ends, and whitespace begins
            n_fontarea = n_fontarea + 1  # so, it's the end of a n_fontarea (+1)

    if debug:
        # Tell me how many font areas there are (i.e. how many vertical lines of text)
        print("\nNumber of font areas = ", n_fontarea)

        # We can also visualize this data in a matplotlib graph
        plt.plot(cols)
        plt.show()
        
    return n_fontarea, cols



def binarize_nonzero_values(cols, cut_thres = 3, debug = 0):
    '''
    Takes an array of column/row non-zero values from a previously scanned image and binarizes it.
    i.e. If there is data in that column/row that is above our ignore threshold, set its value to 1. All other values are 0.

    Parameters:
        cols (array): An np.array image with column/row data values.
        cut_thres (int): Ignore any values below the cut threshold to avoid cutting lines with only a single pixel or two.
        debug (int): Boolean value, whether or not to print sample images to visualize the process (default = 0).

    Returns:
        bcols (array): The binarized array of column/row data
    '''
    
    # cut_thres (int) - some images produce errors with individual columns of 1 or 2 pixels on their own.
    # We don't want to cut these, so ignore anything below a certain threshold.

    # Make a copy of the cols to manipulate it
    bcols = cols.copy()

    # Return either a 1 if data exists, otherwise it remains 0
    for i in range(len(cols)):
        if cols[i] > 0:  # if some data exists in this col
            if cols[i] > cut_thres:
                bcols[i] = 1  # then set bcols at the same location to 1 (binary)
            else:
                bcols[i] = 0 # set anything below our threshold to 0
    
    if debug:
        print('Original column data:\n', cols)
        print('\nBinarized column data:\n', bcols)
        
    return bcols



##=============== Full Preprocessing Steps ===============##

import cutters

def preprocess_image(img, name = '', scan_direction = 0, cut_thres = 3, debug = 0, return_non_bin_arr = 0):
    '''
    Create a binary array for the width OR height of a given image (based on 
    scan_direction where 1 = height (scan every row), 0 = width (scan every column)).
    
    Parameters:
        img (np.array): The numpy image array to scan.
        name (string): Filename of the image.
        scan_direction (int): Whether to scan horizontally (0 = width, scanning columns)
            or vertically (1 = height, scanning rows).
        cut_thres (int): Ignore any values below the cut threshold to avoid cutting lines with only a single pixel or two
        debug (int): Whether or not to print array values (3 prints extra info).
        return_non_binary_array (int): Whether or not to return the non-binarized form of the array. Default = 0.
        
    Returns:
        arr_tuple (tuple): Tuple contining the:
            1. (array), binarized, that represents image column/row cut points (whitespace = 0, data = 1)
                OR array of values that represents the pixel depth of each column/row (returnNonBinaryArray == 1)
            2. (int) for the number of font areas detected in the image
    '''
    
    # Create binary version of the image.
    ret, bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Create binary inverse (white text on a black background).
    ibin_img = cv2.bitwise_not(bin_img)
    
    # Get the shape of the image so we know how far to scan.
    height, width = bin_img.shape

    # Create Binary Array (function def below)
    arr_tuple = create_bin_arr(ibin_img, name, scan_direction, cut_thres, debug, return_non_bin_arr)
    
    cut_points = cutters.find_cut_points(arr_tuple[0])
    
    return arr_tuple, cut_points



def create_bin_arr(img, name = '', scan_direction = 0, cut_thres = 3, debug = 0, return_non_bin_arr = 0):
    '''
    Create a binary array (all 1s and 0s) from the given args.
    
    Parameters:
        img (np.array): The image we are processing.
        name (string): Filename of the image.
        scan_direction (int): 0 = width, scanning columns left to right | 1 = height, scanning rows top to bottom.
        debug (int): Whether or not to print everything (if debug == 3, it prints a graph of the data).
        return_non_bin_arr (int): Whether or not to return the non-binarized form of the array. Default = 0.
    
    Returns:
        bin_arr (array): Binary array that represents image column/row cut points (whitespace = 0, data = 1).
            OR array of values that represents the pixel depth of each column/row (return_non_bin_arr == 1).
    '''
    
    # Create array of 0s to receive processed values from the image data
    if scan_direction == 0: # width
        zero_arr = np.zeros(img.shape[1])
    elif scan_direction == 1: # height
        zero_arr = np.zeros(img.shape[0])
    else:
        print("Error in create_bin_arr(). Incorrect scan direction.")
        return
    
    # Debugging ----------
    if debug == 1:
        print('Image array length = ', len(zero_arr))
        print('\nBefore scanning:\n', zero_arr)

    # For every col, find anything with data (a pixel of the script)
    for i in range(0, len(zero_arr)):
        if scan_direction == 0:
            zero_arr[i] = cv2.countNonZero(img[:,i])
        elif scan_direction == 1:
            zero_arr[i] = cv2.countNonZero(img[i,:])
    
    # Debugging ----------
    if debug == 1:
        print('\nAfter scanning:\n', zero_arr)
    if debug == 3:
        print('Plot of data points (0s indicate a cut line)')
        # And take a look at the cutXpoints in a graph
        plt.plot(zero_arr)
        plt.show()
        
    # In the case of finding letters, we want to return the non-binary array in order to
    # to a bit more pre-processing, like finding all the valleys (lowest values) in this array.
    if return_non_bin_arr == 1:
        return zero_arr
    
    # Make a copy of the rows to manipulate it
    bin_arr = zero_arr.copy()

    '''
    # Kind of like using ReLU (Rectified Linear Unit) to return either a 1 if data exists, otherwise it remains 0
    for i in range(0, len(zero_arr)):
        if zero_arr[i] > 0:  # if some data exists in this row
            bin_arr[i] = 1  # then set brows at the same location to 1 (binary)
    '''
    ## With `CUT_THRES`   
    # Return either a 1 if data exists, otherwise it remains 0
    for i in range(len(zero_arr)):
        if zero_arr[i] > 0:  # if some data exists in this col
            if zero_arr[i] > cut_thres:
                bin_arr[i] = 1  # then set bcols at the same location to 1 (binary)
            else:
                bin_arr[i] = 0 # set anything below our threshold to 0
                
    ## Re-find `N_FONTAREA`
    # Var to hold num of font areas (vertical lines of text) based on the script
    n_fontarea = 0

    # Determine font areas by checking where a non-zero col ends and a zero col begins
    for i in range(0, (len(bin_arr) - 1)):
        if bin_arr[i] > 0 and bin_arr[i+1] == 0:  # here, our script ends, and whitespace begins
            n_fontarea = n_fontarea + 1  # so, it's the end of a n_fontarea (+1)

    # Tell me how many font areas there are (i.e. how many vertical lines of text)
    print(f"Font areas found in '{name}': {n_fontarea}") 
            
    # if debug == 3:
        # print(bin_arr)

    return bin_arr, n_fontarea



def clean_noise(img, kernel_size = 3, debug = 0):
    '''
    Attempts to clean the noise present in script images with erosion and dilation.
    
    Parameters:
        img (np.array): The image we are removing noise from.
        kernel_size (int): Size of the kernel to use in erosion and dilation.
        debug (int): Boolean value, whether or not to print sample images to visualize the process (default = 0).
    
    Returns:
        img_new (np.array): The cleaned image file.
    '''
    
    # Taking a matrix of kernel_size as the kernel
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # The first parameter is the original image,
    # kernel is the matrix with which image is
    # convolved and third parameter is the number
    # of iterations, which will determine how much
    # you want to erode/dilate a given image.
    img_erosion = cv2.erode(img, kernel, iterations = 1)
    img_dilation = cv2.dilate(img, kernel, iterations = 1)
    img_new = cv2.dilate(img_erosion, kernel, iterations = 1)

    if debug:
        # Visualize the test image (both binary and inverse binary)
        f, axes = plt.subplots(2, 2, figsize = (20, 20))
        axes[0,0].set_title("Input image")
        axes[0,0].imshow(img, cmap = plt.get_cmap('gray'))
        axes[0,1].set_title("Eroded image")
        axes[0,1].imshow(img_erosion, cmap = plt.get_cmap('gray'))
        axes[1,0].set_title("Dilated image")
        axes[1,0].imshow(img_dilation, cmap = plt.get_cmap('gray'))
        axes[1,1].set_title("Cleaned image")
        axes[1,1].imshow(img_new, cmap = plt.get_cmap('gray'))
        plt.show()

    return img_new