# 0612quiz.py
import cv2
import numpy as np
path = './data/'
src   = cv2.imread(path+'manchu01.jpg', cv2.IMREAD_GRAYSCALE)
# tmp_K   = cv2.imread('./data/man_k.jpg', cv2.IMREAD_GRAYSCALE)
src_inv = cv2.bitwise_not(src)
roi = cv2.selectROI(src_inv)
tmp_K = src[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
dst  = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)  # 출력 표시 영상
cv2.imshow('src',  dst)
#1
R1 = cv2.matchTemplate(src, tmp_K, cv2.TM_SQDIFF_NORMED)
minVal, _, minLoc, _ = cv2.minMaxLoc(R1)
print('TM_SQDIFF_NORMED:', minVal, minLoc)

w, h = tmp_K.shape[:2]
bsrc = src.copy()
count = 0
outfilename = input('unitname : ')
for i in range(100):
    R1 = cv2.matchTemplate(bsrc, tmp_K, cv2.TM_SQDIFF_NORMED)
    minVal, _, minLoc, _ = cv2.minMaxLoc(R1)
    print(minVal)
    if minVal > 0.05:
        break
    croi = bsrc[minLoc[0]:minLoc[0]+h, minLoc[1]:minLoc[1]+w]
    id = outfilename + "_unit%d.jpg" % i
    # outfilename = filename[0:flen-3] + id
    outfilename = path +  id
    print(outfilename)
    print()
    # cv2.imwrite(outfilename,croi)
    cv2.rectangle(bsrc, minLoc, (minLoc[0]+h, minLoc[1]+w), (255, 0, 0), -1)
    cv2.rectangle(dst, minLoc, (minLoc[0]+h, minLoc[1]+w), (255, 0, 0), 2)
    count = count + 1

print('count = ', count)
cv2.imshow('dst',  dst)
cv2.imwrite(path+'manchu01_fnd.jpg',dst)
cv2.waitKey()
cv2.destroyAllWindows()
