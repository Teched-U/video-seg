import cv2
import numpy as np
import pytesseract
from pytesseract import Output

#  Text Segmentation
def detect_paragraph(image):
    # img: grayscale
    # Load image, grayscale, Gaussian blur, Otsu's threshold
    # image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Create rectangular structuring element and dilate
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6,6))
    dilate = cv2.dilate(thresh, kernel, iterations=10)

    # Find contours and draw rectangle
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
    
    #cv2.imshow('thresh', thresh)
    #cv2.imshow('dilate', dilate)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 600, 600)
    cv2.imshow('image', image)

    cv2.waitKey()
    return cnts

def crop_image(im, rect, scale):
    xmin, ymin, xmax, ymax = rect
    cropped = im[ymin:ymax, xmin:xmax]
    return cropped

def contours_text(orig, contours):
    for cnt in contours: 
        x, y, w, h = cv2.boundingRect(cnt)


        # Cropping the text block for giving input to OCR 
        cropped = orig[y:y + h, x:x + w] 
        print(y, x)

        # Apply OCR on the cropped image 
        config = ('-l eng --oem 1 --psm 3')
        text = pytesseract.image_to_string(cropped, config=config) 
        data = pytesseract.image_to_data(cropped, config = config)


        print(text)
        print(data)
        print()

#image = cv2.imread("/Users/yizhizhang/Downloads/test_frames/46.jpg")
image = cv2.imread("/Users/yizhizhang/Desktop/andrew.png")
cnts = detect_paragraph(image)
contours_text(image, cnts)

