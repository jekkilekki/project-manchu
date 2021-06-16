# 0502.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
"""
Let's break this down, almost like using functions to perform each task.
(We can turn these bits into functions later if desired.)

1. Divide the source image (scriptSrc) into columns (lines) of text
    - Store an array of the cutXpoints (ints)
    - Store an array of new line imgs (scriptLines)
    - Write each line to a separate file ('./img/lines/manchu01-{line#}.jpg')

2. For each line (scriptLines), Divide the lines of text into words
    - Store an array of the cutYpoints (ints) for each line
    - Store an array of new word imgs (scriptWords)
    - Write each word to a separate file ('./img/words/manchu01-{line#}-{word#}.jpg')

3. For each word (scriptWords), Divide the words into individual letters
    - To do so, we may need to average the surrounding binary pixels
      to find the valleys - we cut the words at the narrowest point (valleys)
    - Store an array of the cutYWpoints (ints) for each word
    - Store an array of new letter imgs (scriptLetters)
    - Write each letter to a separate file ('./img/letters/manchu01-{line#}-{word#}-{letter#}.jpg')
"""

# Manchurian script image source
scriptSrc = cv2.imread('./img/manchu01.jpg', cv2.IMREAD_GRAYSCALE)

# Actually, let's create some variables HERE that we can use later
scriptLines = []
scriptWords = []
scriptLetters = []
"""
1. findLines : Divide the image into columns of text
"""
# cv2.imshow('src', src)
height, width = scriptSrc.shape
print("IMAGE: width = ", width, "height = ", height)

# Create an array with the data from the cols of the image
cols = np.full(width, 0)
# Create binary image (only 1s and 0s) using threshold
ret, bin = cv2.threshold(scriptSrc, 0, 255,
                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# inverse binary image (black bg, white txt)
ibin = cv2.bitwise_not(bin)

# For every col, find anything with data (a pixel of the script)
for i in range(width):
    cols[i] = cv2.countNonZero(ibin[:, i])

# Var to hold num of font areas (vertical lines of text) based on the script
n_fontarea = 0

# Determine font areas by checking where a non-zero col ends and a zero col begins
for i in range(width - 1):
    if cols[i] > 0 and cols[
            i + 1] == 0:  # here, our script ends, and whitespace begins
        n_fontarea = n_fontarea + 1  # so, it's the end of a n_fontarea (+1)

# Tell me how many font areas there are (i.e. how many vertical lines of text)
print("Number of font areas = ", n_fontarea)

# Make a copy of the cols to manipulate it
bcols = cols.copy()

# Kind of like using ReLU (Rectified Linear Unit) to return either a 1 if data exists, or a 0 if no data exists
for i in range(width):
    if cols[i] > 0:  # if some data exists in this col
        bcols[i] = 1  # then set bcols at the same location to 1 (binary)

# Setup image cut points (x axis value) for +1 greater than the number of fontareas
# so that we can cut AROUND each column of text.
# i.e. 13 columns of text requires 14 lines (cut points) to divide them
cutXpoints = np.full(n_fontarea + 1, 0)
# print('cutXpoints = ', cutXpoints) # cutXpoints =  [0 0 0 0 0 0 0 0 0 0 0 0 0 0]

# Initialize variables
j = 0  # cutXpoints counter (we have 14 cut points)
startpoint = 0  # start at the beginning of the image (col 0)

# Loop to determine and set our cutXpoints (where to cut the image for each column of text)
for i in range(1, width):  # start at 1, end at width
    # The first case is the END of the image (width - 1) i.e. if 458 == 458
    if i == width - 1:
        endpoint = i
        cutXpoints[j] = (
            startpoint + endpoint
        ) // 2  # / is floating point division, // is integer division (floor - rounding down)
    # Case 2 is the START of a cut point, i.e. the first col is all 0s (whitespace) and the second col is 1 (script)
    elif bcols[i - 1] == 0 and bcols[i] == 1:
        endpoint = i - 1  # don't cut off the script, cut outside it
        cutXpoints[j] = (startpoint + endpoint) // 2
        j = j + 1  # increment cutXpoints counter
    # Case 3 is the END of a cut point, i.e. the first col has script (1), and the second col is all 0s (whitespace)
    elif bcols[i - 1] == 1 and bcols[i] == 0:
        startpoint = i  # in this case, adjust the startpoint to the current column

# Confirm our points
print("cutXpoints = ", cutXpoints)
# plt.plot(cols)
# plt.show()

# Now, using the cutXpoints we determined, cut out and display one column of text (change array values)
for i in range(0, len(cutXpoints) - 1):
    # print('cutXpoint #', i)
    cutline = bin[0:height, cutXpoints[i]:cutXpoints[i + 1]]
    scriptLines.append(cutline)
    # print('writing img', i)
    filename = './img/lines/manchu01-' + str(i) + '.jpg'
    cv2.imwrite(filename, cutline)
    # print('finished img', i)

print('Number of Lines cut: ', len(scriptLines))
"""
2. findWords : Divide the columns of text into words
"""
for line in range(0, len(scriptLines)):
    rows = np.full(height, 0)
    iline = cv2.bitwise_not(scriptLines[line])

    for i in range(height):
        rows[i] = cv2.countNonZero(iline[i, :])

    # Var to hold num of word areas in column 1
    n_wordarea = 0

    for i in range(height - 1):
        if rows[i] > 0 and rows[i + 1] == 0:
            n_wordarea = n_wordarea + 1

    print("Number of words in line ", line, " = ", n_wordarea)

    brows = rows.copy()

    for i in range(height):
        if rows[i] > 0:
            brows[i] = 1

    cutYpoints = np.full(n_wordarea + 1, 0)

    j = 0
    startpoint = 0

    for i in range(1, height):
        if i == height - 1:
            endpoint = i
            cutYpoints[j] = (startpoint + endpoint) // 2
        elif brows[i - 1] == 0 and brows[i] == 1:
            endpoint = i - 1
            cutYpoints[j] = (startpoint + endpoint) // 2
            j = j + 1
        elif brows[i - 1] == 1 and brows[i] == 0:
            startpoint = i

    print("cutYpoints in line ", line, " = ", cutYpoints)
    # plt.plot(rows)
    # plt.show()

    wordsInLine = []
    for i in range(0, len(cutYpoints) - 1):
        cutword = scriptLines[line][cutYpoints[i]:cutYpoints[i + 1], 0:width]
        wordsInLine.append(cutword)
        filename = './img/words/manchu01-' + str(line) + '-' + str(i) + '.jpg'
        cv2.imwrite(filename, cutword)

        scriptWords.append(cutword)

print('Number of Words cut: ', len(scriptWords))
"""
3. findLetters() : Divide the words into individual letters
"""
for word in scriptWords:
    height, width = word.shape
    print("WORD: width = ", width, "height = ", height)

    # cv2.imshow('Letter finding word', word)
    w_rows = np.full(height, 0)

    ret, w_bin = cv2.threshold(word, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    w_ibin = cv2.bitwise_not(w_bin)

    for i in range(height):
        w_rows[i] = cv2.countNonZero(w_ibin[i, :])

    n_letters = 0

    print("wordRows = ", w_rows)

    # Use Gradient Descent Momentum to solve this?
    # Step = Average of previous steps, beta is some constant between 0 & 1 that we can multiple each step by
    # Step(n) = Step(n) + beta*Step(n-1) + beta^2*Step(n-2) + beta^3*Step(n-3)... etc.
    min = w_rows[0]  # initial low value
    max = w_rows[0]  # initial high value

    for i in range(2, height - 1):
        # add two sibling values
        sum0 = w_rows[i - 2] + w_rows[i - 1]
        sum1 = w_rows[i - 1] + w_rows[i]
        sum2 = w_rows[i] + w_rows[i + 1]
        valley = 0
        # when sum is rising, cut
        if sum2 > sum1 and sum1 < sum0:
            valley += 1
            # if w_rows[i] != 0 and w_rows[i] > 1:
            #     min = w_rows[i]
            #     if w_rows[i] > min:
            #         max = w_rows[i]
            # if w_rows[i-1] >= 4 and w_rows[i] <= 4 and w_rows[i+1] >= 4:
            print("checking: ", w_rows[i - 1], ", ", w_rows[i], ", ",
                  w_rows[i + 1])
            n_letters = n_letters + 1
    print('min = ', min, 'max = ', max)

    print("Number of letters in word = ", n_letters)

    arows = w_rows.copy()

    for i in range(height):
        if w_rows[i] > 0:
            arows[i] = 1

    cutSubYpoints = np.full(n_letters + 1, 0)

    j = 0
    startpoint = 0

    # for i in range(1, height):
    #     if i == height - 1:
    #         endpoint = i
    #         cutSubYpoints[j] = (startpoint + endpoint) // 2
    #     elif arows[i - 1] == 0 and arows[i] == 1:
    #         endpoint = i - 1
    #         cutSubYpoints[j] = (startpoint + endpoint) // 2
    #         j = j + 1
    #     elif arows[i - 1] == 1 and arows[i] == 0:
    #         startpoint = i

    print("cutSubYpoints = ", cutSubYpoints)
    plt.plot(w_rows)
    plt.show()

    # Create a new image with the cut points
    # for i in range(0, len(cutSubYpoints) - 1):
    #     fletter = bin[cutSubYpoints[i]:cutSubYpoints[i + 1], 0:width]
    #     cv2.imshow('letter', fletter)

# Determine word areas by checking where a non-zero col ends and a zero col begins
# for i in range(height - 1):
# if

# So, now that we have columns of text, lets find cutYpoints for every word
# let's start with the first column of text (and build up later)
# cutYpoints =

# for i in range(width):
#    print(i, " = ", cols[i])

# # print([rows])
# roi = cv2.selectROI(src)
# print('roi = ', roi)
# bimg = src[roi[1]:roi[1]+roi[3],roi[0]:roi[0]+roi[2]]
# ret, dst = cv2.threshold(src, 0, 255,
#                             cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# cv2.imshow('dst Otsu+Binary',  dst)

# dst2 = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
#                            cv2.THRESH_BINARY, 51, 7)
# cv2.imshow('dst2 AdaptiveThreshMeanC+Binary',  dst2)

# dst3 = cv2.adaptiveThreshold(bimg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                            cv2.THRESH_BINARY, 51, 7)
# cv2.imshow('dst3 AdaptiveThreshGaussianC+Binary',  dst3)
# cv2.imwrite('./img/manchu01b.jpg',dst3)

cv2.waitKey()
cv2.destroyAllWindows()
