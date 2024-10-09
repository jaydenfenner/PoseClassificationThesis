import matplotlib.pyplot as plt
from inferenceUtils.loadImage import readDepthPngFromSimLab

def main():
    subj = 7
    img = readDepthPngFromSimLab(subj=subj)
    cropped, _, _ = cropDepthPngFromSimLab(img, subj=subj)
    displayTestCrop(img, cropped)
    pass

def cropNpImage_scaleY_XY(img, scaleY, shiftX, shiftY):
    ''' Crop Deoth image to a % of max height, and shift by %(x, y)
    - Return (cropped, newMinY, newMinX)
    '''
    origHeight, origWidth = img.shape[:2]
    shiftPercentageYX = [shiftY, shiftX]
    newSize = int(origHeight * scaleY)
    newMinY = (origHeight - newSize) // 2 + int(origHeight * shiftPercentageYX[0])
    newMinX = (origWidth - newSize) // 2 + int(origWidth * shiftPercentageYX[1])
    cropped = img[newMinY:newMinY+newSize, newMinX:newMinX+newSize]
    return cropped, newMinY, newMinX

def cropDepthPngFromSimLab(img, subj):
    # img, newMinY, newMinX = crop.standardPercentageCrop(img) #! 75% standard crop shifted 1% down
    cropped, newMinY, newMinX = individualSubjectCrop(img, subj) #! crop separately for each subject to only bed (exclude bedhead)
    return cropped, newMinY, newMinX

def individualSubjectCrop(img, subj):
    '''crop separately for each subject to only bed (exclude bedhead)'''
    percentagesPerSubj = [ #[heightScale, shiftX, shiftY]
        [0.67, 0., 0.07], # 00001
        [0.66, 0.03, 0.045], # 00002
        [0.68, 0.04, 0.035], # 00003 #! note subj 3 has head above top of bed, not present for other subjects
        [0.67, 0.04, 0.045], # 00004
        [0.67, 0.04, 0.02], # 00005
        [0.67, 0.04, 0.02], # 00006
        [0.67, 0.04, 0.045], # 00007
    ]
    cropParams_scaleY_x_y = percentagesPerSubj[subj-1]
    cropped, newMinY, newMinX = cropNpImage_scaleY_XY(
        img,
        cropParams_scaleY_x_y[0],
        cropParams_scaleY_x_y[1],
        cropParams_scaleY_x_y[2]
    )
    
    return cropped, newMinY, newMinX

def standardPercentageCrop(img):
    '''crop image to fixed square in centre, 75% of original and 1% down'''
    origHeight, origWidth = img.shape[:2]
    shiftPercentageYX = [0.01, 0]
    newSize = int(origHeight * 0.75)
    newMinY = (origHeight - newSize) // 2 + int(origHeight * shiftPercentageYX[0])
    newMinX = (origWidth - newSize) // 2 + int(origWidth * shiftPercentageYX[1])
    img = img[newMinY:newMinY+newSize, newMinX:newMinX+newSize]
    return img, newMinY, newMinX

def displayTestCrop(img, cropped):
    # Create a figure with 2 subplots side by side
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Display original image
    axs[0].imshow(img, cmap='gray')
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    # Display cropped image
    axs[1].imshow(cropped, cmap='gray')
    axs[1].set_title('Cropped Image')
    axs[1].axis('off')

    # Show the images
    plt.show()

if __name__ == '__main__':
    main()