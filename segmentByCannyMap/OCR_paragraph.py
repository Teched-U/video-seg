import numpy as np
import cv2
import time
from ccl import *
from matplotlib import pyplot as plt
import pytesseract
from pytesseract import Output
#from swt import *
import execnet
import click

class Words:
    type = None
    def __init__(self, x, y, width, letters):
        self.x = x
        self.y = y
        self.width = width

    def set_Type(self, type):
        self.type = type

def call_python_version(Version, Module, Function, ArgumentList):
    gw      = execnet.makegateway("popen//python=python%s" % Version)
    channel = gw.remote_exec("""
        from %s import %s as the_function
        channel.send(the_function(*channel.receive()))
    """ % (Module, Function))
    channel.send(ArgumentList)
    return channel.receive()

#  Text Segmentation
def detect_paragraph(image):
    # img: grayscale
    # Load image, grayscale, Gaussian blur, Otsu's threshold
    # image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Create rectangular structuring element and dilate
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    dilate = cv2.dilate(thresh, kernel, iterations=4)

    # Find contours and draw rectangle
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    return cnts
    # for c in cnts:
    #     x,y,w,h = cv2.boundingRect(c)
    #     cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)

    #cv2.imshow('thresh', thresh)
    #cv2.imshow('dilate', dilate)
    # cv2.imshow('image', image)
    # cv2.waitKey()

def contours_text(orig, contours):
    words_objects = []
    for cnt in contours: 
        x, y, w, h = cv2.boundingRect(cnt)

        # Drawing a rectangle on copied image 
        #rect = cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 255, 255), 2) 
        
        #cv2.imshow('cnt',rect)
        #cv2.waitKey()

        # Cropping the text block for giving input to OCR 
        cropped = orig[y:y + h, x:x + w]
        width = call_python_version("2.7", "swt", "getWidthOfLetters", [cropped])

        # Apply OCR on the cropped image 
        config = ('-l eng --oem 1 --psm 3')
        text = pytesseract.image_to_string(cropped, config=config) 
        #data = pytesseract.image_to_data(cropped, config=config, output_type=Output.DICT)
        if (len(text) > 0):
            words = create_text_objects(text, x, y, width)
            words_objects.append(words)
    content_analysis(words_objects)

def create_text_objects(text, x, y, width):
    # x, y, width, letters
    words = Words(x, y, width, text)

def content_analysis(objects):
    # sort text objects by their heights (ascending order)
    sorted_objects = sorted(objects, key=lambda x: x.y, reverse=False)
    diff = 0
    index = 0
    # find title objects
    # while height difference exceeds 30?
    while (diff <= 30):
        if (if_title(sorted_objects[index])):
            print("!!!Title Here")
            print(sorted_objects[index].text)
            sorted_objects[index].set_Type("Title")
            diff = abs(sorted_objects[index + 1] - sorted_objects[index])
        index += 1
    # width mean besides title
    not_titles = filter(lambda c: c.type != "Title", objects)
    s_mean = np.mean([c.width for c in not_titles])
    y_mean = np.mean([c.y for c in not_titles])
    x_mean = np.mean([c.x for c in not_titles])
    y_max = sorted_objects[-1].y
    for object in not_titles:
        define_content_type(object, s_mean, x_mean, y_mean, y_max)


def if_title(object, frame):
    """
    the text line is in the upper third part of the frame
    the text line has more than three characters
    the horizontal start position of the text line is not larger than the half of the frame width,
    one of three highest text lines
    """
    width, height = frame.shape
    if (object.y <= height / 3):
        if (len(object.text) > 3):
            if (object.x < width / 2):
                return True
    return False

def define_content_type(object, s_mean, x_mean, y_mean, y_max):
    # subtitle/key point if st > smean ^ x < xmean
    if (object.width > s_mean):
        object.set_Type("Subtitle")
    elif (object.width < s_mean & object.y > y_mean & abs(y_max - object.y) <= 10):
        object.set_Type("Footline")
    else:
        object.set_Type("Normal")


def get_Canny(pathIn):
    # Canny maps
    frames = []
    # imgs
    imgs = []
    vid_cap = cv2.VideoCapture(pathIn)
          
    if vid_cap.isOpened():
        # get rate
        rate=vid_cap.get(5)
        # get frame number
        FrameNumber=vid_cap.get(7)
        duration=FrameNumber/rate
        # get duration
        print(duration)
    success, image = vid_cap.read()
    count = 0
    while success:
        temp = vid_cap.get(0)
        cv2.imwrite("/Users/yizhizhang/Downloads/test_frames/" + str(count) + ".jpg", image)  # save frame as JPEG file
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(gray,50,150)
        frames.append(canny)
        imgs.append(image)
        count += 1
        vid_cap.set(cv2.CAP_PROP_POS_MSEC, 1 * 1000 * count)
        success, image = vid_cap.read()
        if temp == vid_cap.get(0):
            print("loop ends")
            break
        print('Total frames: ', count)

    # When everything done, release the capture
    vid_cap.release()
    cv2.destroyAllWindows()
    return frames, duration, imgs

def get_Differential(frames):
    differentials = []
    for i in range(1,len(frames)):
        differentials.append(frames[i] - frames[i-1])
    return differentials



  
  


def main(pathin):
    #pathin = "/Users/yizhizhang/Downloads/test.mp4"
    frames, duration, imgs = get_Canny(pathin)
    differentials = get_Differential(frames)
    CCs = []
    for differential in differentials:
        #change to bool image
        arr = np.asarray(differential)
        arr = arr != 255
        # CC Analysis
        result = connected_component_labelling(arr, 4)
        CCs.append(np.max(result))
        print(np.max(result))
    # Threshold processing (20)
    CCs = np.array(CCs)
    tmp = np.where(CCs >20)
    # Time stamps of key_frames after first segmentation
    index = tmp[0] + 1
    # Get key_frames (first step)
    key_frames = imgs[index]




def main(image):
    #image = cv2.imread("/Users/yizhizhang/Desktop/andrew.png")
    image = cv2.imread(image)
    cnts = detect_paragraph(image)
    contours_text(image, cnts)


@click.command()
@click.option(
    "-i",
    "image",
    default="/Users/yizhizhang/Downloads/test_frames/46.jpg",
    help="Image to process"
)
def _main(image: str):
    main(image)
    

if __name__ == "__main__": 
    _main()
