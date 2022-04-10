import cv2
import helpers

# Define a function to find cutPoints for any given array.
# (This can be used for both horizontal and vertical cutPoints)
def find_cut_points(bcols, debug = 0):
    '''
    Find the points to cut the image.
    
    Parameters:
        bcols (array): A binary array (1s and 0s) equivalent to the side of the image in question.
        debug (int): int value, whether or not to print all the output.
        
    Returns:
        cut_points (array): An array of tuples for every cutting Start and End point
    
    '''
    
    cut_points = []

    # Initialize variables
    startpoint = 0  # start at the beginning of the image (col 0)
    endpoint = 0
    hasPair = 0
    lastItem = len(bcols)

    # Loop to determine and set our cutXpoints (where to cut the image for each column of text)
    for i in range(0, len(bcols) - 1):

        if endpoint == 1:
            cut_points.append([startpoint, i])
            endpoint = 0
            hasPair = 1
            if debug == 1:
                print('cut_points end: ', i)
        if bcols[i] == 0 and bcols[i+1] == 1: # This is the START of a script line
            startpoint = i
            hasPair = 0
            if debug == 1:
                print('\ncut_point start: ', i)
        if bcols[i] == 1 and bcols[i+1] == 0: # This is the END of a script line
            endpoint = 1
    
    # Edge case found mostly in letters where the last endpoint sometimes is not found.
    if startpoint > endpoint and hasPair == 0:
        cut_points.append([startpoint, len(bcols)])
        
    # Confirm our points
    if debug == 1:
        print("\ncut_points = ", cut_points)
        print("\nNum slices based on cut_points = ", len(cut_points))
    
    return cut_points



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