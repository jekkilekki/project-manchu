import os
import cv2

# Load all files in given folder
def load_images_from_folder(path, flag = 0, debug = 0):
    '''
    Loads images from a given folder path and returns an array of the images.

    Parameters:
        path (str): The path to the desired directory
        flag (int): The cv2 mode to read the image (0 = cv2.IMREAD_GRAYSCALE, etc)
        debug (int): A bool, if enabled, that prints the first image in the array to check that data is loaded

    Returns:
        images (array): An array of images in the containing directory
    '''
    images = []
    filenames = []
    
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path, filename), flag)
        if img is not None:
            images.append(img)
            filenames.append(filename)
            
    if images:
        # Debugging
        if debug == 1:
            visualize_dataset(images)
            
    else:
        print('No images found in that directory. Please check the path again.')

    return images, filenames



def visualize_dataset(dataset):
    if dataset:
        print('Debugging & Visualization:')

        print('\nDataset size: {}'.format(len(dataset)))
        print('\nImages sizes for first 5 images:')
        for i in range(0,5):
            print('{}: {}'.format(i, dataset[i].shape))

        im = random.randint(0,5)
        print('\nVisualizing Image {}:'.format(im))
        plt.imshow(dataset[im], cmap = plt.get_cmap('gray'))
        
        
        
def make_folder(path, subfolder = ""):
    '''
    Creates a new folder if one doesn't exist, and returns the path of the folder that exists.

    Parameters:
        path (str): The path to the desired directory
        subfolder (str): A subfolder of the desired directory (default = "")

    Returns:
        path (str): The path to the directory (including subfolder if given)
    '''
    path = os.path.join(path, subfolder)
    
    # Check if our Cut folder exists - if not, create it
    if not os.path.exists(path):
        os.makedirs(path)
        print("The new directory is created!")
    else:
        print("That directory already exists. Nothing created.")
        
    return path



