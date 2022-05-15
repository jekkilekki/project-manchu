import cv2
import numpy as np
import matplotlib.pyplot as plt
# import helpers

"""
Cutting Steps:

1. SINGLE Image Processing
    1. Count & binarize nonzeroes (in inverse-binary image)
    2. Estimate cut points
    3. Fix cut points (& visualize if debug) - SAVE
2. MULTIPLE Image Processing
"""

def count_nonzeros(img, dim = 1, debug = 0):
    '''
    Locates font areas in a (non-inversed) binarized script image.
    We DON'T need a cut threshold yet - this is just finding POSSIBLE font areas
    and returning column values (which we then compare later to find true font areas).

    Parameters:
        img (array): An np.array image
        dim (int): Which dimension to consider (height = 0, width = 1) Default: width (find lines)
        debug (int): Boolean value, whether or not to print sample images to visualize the process (default = 0)

    Returns:
        n_fontarea (int): The number of font areas (lines of script, or individual words) that were found in the image
        cols (array): An array with the non-zero values determined for each column in the image
    '''
    
    # Var to hold num of font areas (vertical lines of text) based on the script
    n_fontarea = 0
    height, width = img.shape
    ibin = cv2.bitwise_not(img)
    
    # Create an array of 0s to represent the entire dimension of the image (459px or etc)
    if dim == 0:
        cols = np.full(height, 0)
    if dim == 1:
        cols = np.full(width, 0)
    
    if debug:
        print('Image dimension = ', len(cols))
        print('\nBefore scanning:\n', cols)

    # For every col in the array, find anything with data (a non-zero value, i.e. a pixel of the script)
    if dim == 0:
        for i in range(height):
            cols[i] = cv2.countNonZero(ibin[i, :])
    if dim == 1:
        for i in range(width):
            cols[i] = cv2.countNonZero(ibin[:, i])
    
    if debug:
        print('\nAfter scanning:\n', cols)

    # Determine font areas by checking where a non-zero col ends and a zero col begins
    if dim == 0:
        for i in range(height - 1):
            if cols[i] > 0 and cols[i+1] == 0: # here, script ends, whitespace begins
                n_fontarea = n_fontarea + 1
    if dim == 1:
        for i in range(width - 1):
            if cols[i] > 0 and cols[i+1] == 0: # here, our script ends, and whitespace begins
                n_fontarea = n_fontarea + 1  # so, it's the end of a n_fontarea (+1)

    if debug:
        # Tell me how many font areas there are (i.e. how many vertical lines of text)
        print("\nPossible font areas = ", n_fontarea)

        # We can also visualize this data in a matplotlib graph
        plt.plot(cols)
        plt.show()
        
    return n_fontarea, cols



# Define a function to find cutPoints for any given array.
# (This can be used for both horizontal and vertical cutPoints)
def estimate_cut_points(cols, debug = 0):
    '''
    Estimate the points to cut the image. These may not be exact, so the next function will improve them.
    
    Parameters:
        cols (array): An array of pixel counts in each column / row of the side of the image in question.
        debug (int): int value, whether or not to print all the output.
        
    Returns:
        cut_points (array): An array of tuples for every cutting Start and End point
    
    '''
    
    cut_points = []

    # Initialize variables
    startpoint = 0  # start at the beginning of the image (col 0)
    endpoint = 0
    hasPair = 0
    lastItem = len(cols)

    # Loop to determine and set our cutXpoints (where to cut the image for each column of text)
    for i in range(0, len(cols) - 1):

        if endpoint == 1:
            cut_points.append([startpoint, i])
            endpoint = 0
            hasPair = 1
            if debug == 1:
                print('cut_points end: ', i)
        if cols[i] == 0 and cols[i+1] > 0: # This is the START of a script line
            startpoint = i
            hasPair = 0
            if debug == 1:
                print('\ncut_point start: ', i)
        if cols[i] > 0 and cols[i+1] == 0: # This is the END of a script line
            endpoint = 1
    
    # Edge case found mostly in letters where the last endpoint sometimes is not found.
    if startpoint > endpoint and hasPair == 0:
        cut_points.append([startpoint, len(cols)])
        
    # Confirm our points
    if debug == 1:
        print("\ncut_points = ", cut_points)
        print("\nNum slices based on cut_points = ", len(cut_points))
    
    return cut_points



def fix_cut_points(cut_points, threshold = 20, debug = 0, img = []):
    
    if debug:
        # convert to color
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
    true_cut_pts = []

    for pt in cut_points:
        left = pt[0]
        right = pt[1]
        
        if right - left < threshold:
            continue
            
        if debug:
            img = cv2.line(img, (pt[0], 0), (pt[0], img.shape[0]), (0, 0, 255), 2) # bgr
            img = cv2.line(img, (pt[1], 0), (pt[1], img.shape[0]), (0, 0, 255), 2) # bgr
        
        true_cut_pts.append(pt[0])
        true_cut_pts.append(pt[1])
    
    if debug:
        # display results
        fig = plt.figure(figsize = (15, 30))
        ax1 = fig.add_subplot(1,1,1)
        ax1.imshow(np.array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        plt.show()
    
    return true_cut_pts



def cutpts_multiple(dataset):
    
    dataset_cutpts = []
    font_areas = []
    
    for img in dataset:
        _, cols = count_nonzeros(img)
        cutpts = estimate_cut_points(cols)
        true_cutpts = fix_cut_points(cutpts, threshold = 12)
        
        dataset_cutpts.append(true_cutpts)
        font_areas.append(len(true_cutpts) / 2)
    
    return dataset_cutpts, font_areas
        




def cut_and_save(img, cut_points, cut_type = 0, threshold = 3, save_location = '', filename = '', debug = 0):
    '''
    Cut and Save.
    
    Parameters:
        img (np.array): The image to cut.
        cut_points: An array of tuples [startPoint, endPoint] of cutpoints.
        cut_type: int, 0 = "lines," vertical cut line; 1 = "words," horizontal cut line; 2 = "letters," horizontal cut line.
        threshold: int, ignore anything less than a certain threshold. We don't need nearly empty lines.
        save_location: The folder in which to save the cut up images.
        
    Returns:
        pieces (np.array): An array of all the image pieces that have been cut
    '''
    
    if img is not None:
        
        pieces = []
        height, width = img.shape
        
        # Choose the correct save location
        if cut_type == 0: # "lines"
            path = helpers.make_folder(save_location, 'lines')
        elif cut_type == 1: # "words"
            path = helpers.make_folder(save_location, 'words')
        elif cut_type == 2: # "letters"
            path = helpers.make_folder(save_location, 'letters')
        else: # root save folder
            path = save_location
            
        file = path + '/' + filename[:-4]
        
        # Now, using the cutPoints we determined, cut out and save
        for i, pt in enumerate(cut_points):
            
            ## HERE is our threshold that makes sure our lines are greater than a certain width/height
            if (pt[1] - pt[0]) > threshold:
            
                if cut_type == 0: # vertical cuts, use the full height
                    cut_line = img[0:height, pt[0]:pt[1]]

                elif cut_type == 1 or cut_type == 2: # horizontal cuts, use the full width
                    cut_line = img[pt[0]:pt[1], 0:width]

                pieces.append(cut_line)
                print(f"writing {filename} #{i} to {path}")
                cv2.imwrite((file + '-' + str(i) + '.jpg'), cut_line)
                # print('finished img', i)
        
        if debug: 
            print('Number of Pieces cut: ', len(pieces))
        
        return pieces
    
    

########## DEPRECATED ##########

# def binarize_nonzeros(cols, cut_thres = 3, debug = 0):
#     '''
#     Takes an array of column/row non-zero values from a previously scanned image and binarizes it.
#     i.e. If there is data in that column/row that is above our ignore threshold, set its value to 1. All other values are 0.

#     Parameters:
#         cols (array): An np.array image with column/row data values.
#         cut_thres (int): Ignore any values below the cut threshold to avoid cutting lines with only a single pixel or two.
#         debug (int): Boolean value, whether or not to print sample images to visualize the process (default = 0).

#     Returns:
#         bcols (array): The binarized array of column/row data
#     '''
    
#     # cut_thres (int) - some images produce errors with individual columns of 1 or 2 pixels on their own.
#     # We don't want to cut these, so ignore anything below a certain threshold.

#     # Make a copy of the cols to manipulate it
#     bcols = cols.copy()

#     # Return either a 1 if data exists, otherwise it remains 0
#     for i in range(len(cols)):
#         if cols[i] > 0:  # if some data exists in this col
#             if cols[i] > cut_thres:
#                 bcols[i] = 1  # then set bcols at the same location to 1 (binary)
#             else:
#                 bcols[i] = 0 # set anything below our threshold to 0
    
#     if debug:
#         print('Original column data:\n', cols)
#         print('\nBinarized column data:\n', bcols)
        
#     return bcols