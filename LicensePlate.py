"""
Automated License Plate Segmentation & Recognition System

This application requires the list of packages listed in the notepad requirements.txt,
as well as Tesseract OCR available at the repository https://tesseract-ocr.github.io/tessdoc/Home.html.
For Windows-based system, latest version of Tesseract available is 3.02, available at the link below:
https://tesseract-ocr.github.io/tessdoc/Downloads.html
"""

! pip install -r requirements.txt


import re
import os
import cv2
import glob
import numpy as np
import pytesseract
import matplotlib.pyplot as plt

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"


def plate_detection(file):
    
    plate_file = file.split('\\')[-1]
    print("Processing image", plate_file)
    plate_true.append(os.path.splitext(plate_file)[0]) # Reading the true value of the license plate from the filename
    
    global image, image_gray
    image = cv2.imread(file)
    image_dimy, image_dimx = image.shape[0], image.shape[1] # Saving the dimensions of original image for resizing of image mask
        
        
    for dims in [(500, 300), (400, 240), (300, 180)]:
        image_resz = cv2.resize(image, (dims[0], dims[1]))

        filtr_smth = np.ones((3, 3))/9 # Simple averaging filter
        
        image_gray = cv2.cvtColor(image_resz, cv2.COLOR_BGR2GRAY)
        image_blur = cv2.filter2D(image_gray, -1, filtr_smth)
        image_edge = cv2.Canny(image_blur, 30, 200)
        
        image_grays.append(image_gray)
        image_blurs.append(image_blur)
        image_edges.append(image_edge)

        contour, _ = cv2.findContours(image_edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contour    = sorted(contour, key=cv2.contourArea, reverse=True)[:10]

        plate_detc = None
        for cont in contour:
            peri = cv2.arcLength(cont, True)
            appr = cv2.approxPolyDP(cont, 0.02 * peri, True)
            if len(appr) == 4:
                plate_detc = appr
                break
        try:
            image_mask = np.zeros(image_gray.shape, np.uint8)
            image_cont = cv2.drawContours(image_mask, [plate_detc], 0, 255, -1)
            image_cont = cv2.bitwise_and(image_resz, image_resz, mask=image_mask)
            image_mask = cv2.resize(image_mask, (image_dimx, image_dimy))

            image_conts.append(image_cont)
            image_masks.append(image_mask)
            
            x, y = np.where(image_mask==255)
            global t,l,b,r
            t, l = np.min(x)-5, np.min(y)+15 # identify top and left edges of the bounded region with padding
            b, r = np.max(x)+5, np.max(y)-15 # identify bottom and right edges of the bounded region with padding

            image_crop = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[t:b, l:r]
            break
        except:
            print("License plate not detected. Please check the image.")
            pass

    return image_crop


def plate_recognition(image_crop):
    
    plate_tess = pytesseract.image_to_string(image_crop)
    plate_tess = plate_tess.strip()
    plate_tess = re.sub(r'[^\w]', '', plate_tess)
    
    if plate_tess == '':
        image_crop = image[t:b, l:r]
        plate_tess = pytesseract.image_to_string(image_crop)
        plate_tess = plate_tess.strip()
        plate_tess = re.sub(r'[^\w]', '', plate_tess)
        
    image_crops.append(image_crop)
    plate_pred.append(plate_tess)
    
    return plate_tess


def plate_detection_plot(idx):
    fig, ax = plt.subplots(2, 3, figsize=(16,9), dpi=100)

    ax[0,0].imshow(image_grays[idx], cmap='gray');
    ax[0,0].axes.xaxis.set_visible(False);
    ax[0,0].axes.yaxis.set_visible(False);
    ax[0,0].axes.set_title('Step 1. Grayscale Image');

    ax[0,1].imshow(image_blurs[idx], cmap='gray');
    ax[0,1].axes.xaxis.set_visible(False);
    ax[0,1].axes.yaxis.set_visible(False);
    ax[0,1].axes.set_title('Step 2. Blurred Image');

    ax[0,2].imshow(image_edges[idx], cmap='gray');
    ax[0,2].axes.xaxis.set_visible(False);
    ax[0,2].axes.yaxis.set_visible(False);
    ax[0,2].axes.set_title('Step 3. Image Edge');

    ax[1,0].imshow(image_masks[idx], cmap='gray');
    ax[1,0].axes.xaxis.set_visible(False);
    ax[1,0].axes.yaxis.set_visible(False);
    ax[1,0].axes.set_title('Step 4. Bounding Rectangle');

    ax[1,1].imshow(image_conts[idx]);
    ax[1,1].axes.xaxis.set_visible(False);
    ax[1,1].axes.yaxis.set_visible(False);
    ax[1,1].axes.set_title('Step 5. Masked Image');

    ax[1,2].imshow(image_crops[idx], cmap='gray');
    ax[1,2].axes.xaxis.set_visible(False);
    ax[1,2].axes.yaxis.set_visible(False);
    ax[1,2].axes.set_title('Step 6. Cropped Image');

plate_true = []
plate_pred = []

image_grays = []
image_blurs = []
image_edges = []
image_masks = []
image_conts = []
image_crops = []


def main():
    
    image_paths = glob.glob(os.getcwd()+'/Test Images/*.jpg') + glob.glob(os.getcwd()+'/Test Images/*.png')
      
    for idx, file in enumerate(image_paths):
        plate_detection_plot(idx)
        image_crop = plate_detection(file)
        plate_tess = plate_recognition(image_crop)
        
        print("Detected license plate:", plate_tess)


if __name__ == "__main__":
    main()