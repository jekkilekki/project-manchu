# vertical-text-detect.py

class VertText:
    """
    A class used to represent an Image of vertical text
    Examples: Manchurian script, Japanese, Chinese, etc

    Attributes
    ----------
    num_lines : int
        The number of lines of script the image contains
    num_words : int
        The total number of words in the script
    num_letters : int
        The total number of letters in the words of the script
        (note that this is only necessary in languages where
        the script is connected - like Manchurian script, 
        but not in scripts where the words stand alone
        like Japanese and Chinese, for example)
    arr_lines : np.array
        An array containing subimages of each line of text
    arr_words : np.array
        A two-dimensional array containing each word of each line
        in an index corresponding to the line of the script it
        can be found in
    arr_letters : np.array
        A (three?)-dimensional array containing each letter
        within each word of the script with a corresponding index
        to which word in which line it is found

    Methods
    ----------
    countLines(src)
        Counts the number of vertical lines of text within the
        given src image
    countWords(line)
        Counts the number of words within a given line subimage
    countLetters(word)
        Counts the number of letters within a given word
    findLines(src)
        Subdivides the given src image into an array of subimages
        containing each line of the script
    findWords(line)
        Subdivides the given line image into an array of subimages
        containing each word of the line
    findLetters(word)
        Subdivides the given word image into an array of subimages
        containing each letter of the word
    """

    def __init__(self, src):
        """
        Parameters
        ----------
        src : str
            The path to the image of script text to be read in 
            and manipulated
        """

        self.src = src

    def countLines(src):
        """

        """
    
    def countWords(line):
        """

        """
    
    def countLetters(word):
        """

        """

    def findLines(src):
        """

        """

    def findWords(line):
        """

        """

    def findLetters(word):
        """

        """