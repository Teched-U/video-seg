import numpy as np
import cv2
import time
from .ccl import *
from matplotlib import pyplot as plt
import pytesseract
from pytesseract import Output
from pathlib import Path
import subprocess
import os
import json
from typing import Dict
#from swt import *
import execnet
import click

SLIDE_PATH = '/data/slides'

class Words:
    type = None
    def __init__(self, x, y, width, height, letters):
        self.x = int(x)
        self.y = int(y)
        # stroke width
        self.width = width
        # text height
        self.height = height
        self.text = letters

    def set_Type(self, type):
        self.type = type

    def __str__(self):
        return f"[x={self.x},y={self.y},width={self.width},height={self.height}]\nText:\n{self.text}"

#  Text Segmentation
def detect_paragraph(image):
    # img: grayscale
    # Load image, grayscale, Gaussian blur, Otsu's threshold
    # image = cv2.imread(path)
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(image, (5,5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Create rectangular structuring element and dilate
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    dilate = cv2.dilate(thresh, kernel, iterations=6)

    # Find contours and draw rectangle
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    return cnts
    # for c in cnts:
    #     x,y,w,h = cv2.boundingRect(c)
    #     cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)

    # cv2.imshow('thresh', thresh)
    # cv2.imshow('dilate', dilate)
    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('image', 600, 600)
    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    # assert(0==1)

def filter_ocr_results(d):
    """
    input:
        d : {
            'level': [...]
            'page_num': [...],
            'block_num': [...],
            'par_num': [...],
            'word_num': [...],
            'left': [...],
            'top': [...],
            'width': [...],
            'conf': [...],
            'text': [...],
        }

    return:
        bool: if should be treated as a valid ocr result 
        dict: result dictionary with the same structure as the input.
    """
    CONF_THRESHOLD = 95
    # VALID_WORD_THRESHOLD = 0

    # Filter by individual conf score
    result = {
        'left': [],
        'width': [],
        'top': [],
        'text': [],
        'height': [],
    }

    valid_word = 0
    for idx, conf in enumerate(d['conf']):
        if int(conf) > CONF_THRESHOLD:
            result['text'].append(d['text'][idx])
            result['left'].append(d['left'][idx])
            result['top'].append(d['top'][idx])
            result['height'].append(d['height'][idx])
            valid_word += 1
            

    # if 1.0 * valid_word / len(d['text']) < VALID_WORD_THRESHOLD:
    #     return False, {}
    if valid_word == 0:
        return False, {} 

    return True, result


def analyze_ocr_result(d, x, y):
    """
    Input:
        d: info
        x: bounding box's x
        y: bounding box's y


    Anaylyze the result, output information :
    1. average height
    2. top-left coordinate
    ...
    """

    heights = np.array(d['height'])
    avg_height = np.mean(heights)

    lefts = np.array(d['left'])
    leftest = np.min(lefts)

    tops = np.array(d['top'])
    topest = np.min(tops)

    results = {
        'height': avg_height,
        'top': topest+y,
        'left': leftest+x,
    }
    return results




def contours_text(orig, contours, frame_index):
    count = 0
    words_objects = []
    rows, cols = orig.shape
    for cnt in contours: 
        x, y, w, h = cv2.boundingRect(cnt)

        # Drawing a rectangle on copied image 
        #rect = cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 255, 255), 2) 
        
        #cv2.imshow('cnt',rect)
        #cv2.waitKey()

        # Cropping the text block for giving input to OCR 
        cropped = orig[y:y + h, x:x + w]
        # tmp = np.copy(orig)
        # tmp[0:y,:] = 255
        # tmp[:,0:x] = 255
        # tmp[:,x+w+1:] = 255

        # # check if boundary!!!
        # tmp[y + h + 1:,:] = 255
        #plt.imshow(tmp)
        #plt.show()

        # Apply OCR on the cropped image 
        config = ('-l eng --oem 1 --psm 3')
        text = pytesseract.image_to_string(cropped, config=config) 
        # data = pytesseract.image_to_data(cropped, config=config, output_type=Output.DICT)
        if (len(text) > 0):
            np.save('data'+str(count), cropped) # save
            #new_num_arr = np.load('data.npy') # load
            file = 'data'+str(count)+'.npy'
            
            python3_command = "python2 segmentByCannyMap/swt.py " + file  # launch your python2 script using bash

            process = subprocess.Popen(python3_command.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()
            width = float(output.decode())

            d = pytesseract.image_to_data(cropped, config = config, output_type = Output.DICT)
            n_boxes = len(d['text'])

            # Filtering based on conf score
            valid, d = filter_ocr_results(d)
            
            if not valid:
                continue
            
            result = analyze_ocr_result(d, x, y)
            result['width'] = width

            words = create_text_objects(d, result)
            words_objects.append(words)
    print("begin analysis")
    return content_analysis(words_objects, orig, frame_index)


def create_text_objects(d, result):
    # x, y, width, letters
    words = d['text']
    # TODO(different line words might join together here. Use some other seperators?)
    text = ' '.join(words) 
    words = Words(result['left'], result['top'], result['width'], result['height'], text)

    return words

def merge_slides(data_arr):
    prev_title = ""
    merged_data_arr = []
    for data in data_arr:
        slide = data['result'] 
        cur_title = slide['Title']

        # TODO(word embedding?)
        if prev_title != cur_title: 
            merged_data_arr.append(data)
            prev_title = cur_title

    return merged_data_arr


def content_analysis(objects, frame, frame_index):
    # sort text objects by their heights (ascending order)
    sorted_objects = sorted(objects, key=lambda x: x.y, reverse=False)
    diff = 0
    index = 0
    # find title objects
    # while height difference exceeds 30?
    while diff <= 30:
        if index + 1 == len(sorted_objects):
            break

        if (if_title(sorted_objects[index], frame)):
            print("!!!Title Here:")
            print(sorted_objects[index])

            sorted_objects[index].set_Type("Title")
            diff = abs(sorted_objects[index + 1].y - sorted_objects[index].y)
        index += 1
    # width mean besides title
    #import pdb;pdb.set_trace()
    not_titles = list(filter(lambda c: c.type != "Title", objects))
    #print(len(not_titles))
    s_mean = round(np.mean([c.width for c in not_titles]))
    y_mean = round(np.mean([c.y for c in not_titles]))
    x_mean = round(np.mean([c.x for c in not_titles]))
    y_max = sorted_objects[-1].y
    h_mean = round(np.mean([c.height for c in not_titles]))
    for o in not_titles:
        define_content_type(o, s_mean, x_mean, y_mean, y_max, h_mean)
    #import pdb;pdb.set_trace()
    # 按照y顺序print出title以及subtitle
    data = {}
    data['Title'] = []
    data['Subtitle'] = []
    for object in sorted_objects:
        if (object.type == 'Title'):
            data['Title'].append({
                'text': object.text,
                'x': object.x,
                'y': object.y,
                'height':object.height
            })
        elif (object.type == 'Subtitle'):
            data['Subtitle'].append({
                'text': object.text,
                'x': object.x,
                'y': object.y,
                'height':object.height
            })

    if frame_index is not None:
        with open('data'+str(frame_index)+'.txt', 'w') as outfile:
            json.dump(data, outfile)
    return data

    


def if_title(o, frame):
    """
    the text line is in the upper third part of the frame
    the text line has more than three characters
    the horizontal start position of the text line is not larger than the half of the frame width,
    one of three highest text lines
    """
    width, height = frame.shape
    if (o.y <= height / 3):
        if (len(o.text) > 3):
            #if (object.x < width / 2):
            return True
    return False

def define_content_type(o, s_mean, x_mean, y_mean, y_max, h_mean):
    # subtitle/key point if st > smean ^ ht > h_mean
    if (o.width >= s_mean and o.height >= h_mean):
        o.set_Type("Subtitle")
        print("Subtitle:")
        print(o.text)
    elif (o.width <= s_mean and o.height < h_mean and abs(y_max - o.y) <= 10):
        o.set_Type("Footline")
        print("Footline:")
        print(o.text)
    else:
        o.set_Type("Normal")
        print("Normal:")
        print(o.text)


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
        # cv2.imwrite("/data/test_frames/" + str(count) + ".jpg", image)  # save frame as JPEG file
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

def get_Canny_timestamps(pathIn, start, end):
    STEP_TIME = 3
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

    success, image = vid_cap.read()
    cur_ts = start
    timestamps = []
    video_name_no_suffix = os.path.basename(pathIn).split('.')[0]
    img_dir = Path(os.path.join(SLIDE_PATH, video_name_no_suffix))
    img_dir.mkdir(parents=True, exist_ok=True)
    img_paths = []

    while cur_ts <= end:
        temp = vid_cap.get(0)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(gray,50,150)
        frames.append(canny)
        imgs.append(image)
    
        # Save to image
        img_path = os.path.join(img_dir, f'{cur_ts*1000}.jpg')
        cv2.imwrite(img_path, image)
        img_paths.append(img_path)
        vid_cap.set(cv2.CAP_PROP_POS_MSEC, 1000 * cur_ts)
        success, image = vid_cap.read()
        cur_ts += STEP_TIME
        timestamps.append(cur_ts)

    # When everything done, release the capture
    vid_cap.release()
    cv2.destroyAllWindows()
    return frames, timestamps, imgs, img_paths



def get_Differential(frames):
    differentials = []
    for i in range(1,len(frames)):
        differentials.append(frames[i] - frames[i-1])
    return differentials


def ocr_lib(pathin, start, end):
    """
    Args:
    - pathin: str,  video path 
    - start: float, starting time 
    - end: float, ending time

    Return :
    [
        {
            timestamp:...,
            result: ...,
            slide: ...,
        }
    ]

    """

    # Get Canny
    frames, timestamps, imgs, img_paths = get_Canny_timestamps(pathin, start, end)

    # Get keyframes
    differentials = get_Differential(frames)

    CCs2 = []
    for differential in differentials:
        #change to bool image
        arr = np.asarray(differential)
        arr = arr[0:int(0.2 * differential.shape[1]),:]
        arr = arr != 255
        # CC Analysis
        result = connected_component_labelling(arr, 4)
        CCs2.append(np.max(result))
    CCs2 = np.array(CCs2)
    tmp = np.where(CCs2 >20)[0]
    index2 = np.insert(tmp + 1, 0, 0)
    key_frames_index = index2 

    print(f"Final Key Frame Index: {key_frames_index}")

    # Prepare for OCR
    final_key_frames = [imgs[i] for i in key_frames_index]
    final_timestamps = [timestamps[i] for i in key_frames_index]
    final_img_paths = [img_paths[i] for i in key_frames_index]
    # import pdb;pdb.set_trace()
    #for f in final_key_frames:
    #    plt.imshow(f)
    #    plt.show()
    frame_count = 0

    # frame count + 几秒抽一帧
    data_arr = []
    for frame, ts, img_path in zip(final_key_frames, final_timestamps, final_img_paths):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cnts = detect_paragraph(gray)
        data = contours_text(gray, cnts, key_frames_index[frame_count])
        data_arr.append({
            'timestamp': ts,
            'result': data,
            'slide': img_path
        })
        frame_count += 1


    # Merge similar slides 
    slides = merge_slides(data_arr)

    return slides

  
  


def main(pathin):
    # pathin = "/Users/yizhizhang/Downloads/test3.mp4"
    frames, duration, imgs = get_Canny(pathin)
    differentials = get_Differential(frames)
    # CCs = []
    # for differential in differentials:
    #     #change to bool image
    #     arr = np.asarray(differential)
    #     arr = arr != 255
    #     # CC Analysis
    #     result = connected_component_labelling(arr, 4)
    #     CCs.append(np.max(result))
    #     print(np.max(result))
    # # Threshold processing (20)
    # CCs = np.array(CCs)
    # tmp = np.where(CCs >20)[0]
    tmp = np.array([3, 24])
    # Time stamps of key_frames after first segmentation
    index = np.insert(tmp + 1, 0, 0)
    # import pdb;pdb.set_trace()

    # Get key_frames (first step)
    #import pdb;pdb.set_trace()
    # key_frames = [imgs[i] for i in index]
    # for f in key_frames:
    #     plt.imshow(f)
    #     plt.show()
    
    differentials2 = []
    for i in range(1,len(index)):
        differentials2.append(frames[index[i]] - frames[index[i-1]])

    CCs2 = []
    for differential in differentials2:
        #change to bool image
        arr = np.asarray(differential)
        arr = arr[0:int(0.2 * differential.shape[1]),:]
        arr = arr != 255
        # CC Analysis
        result = connected_component_labelling(arr, 4)
        CCs2.append(np.max(result))
        print(np.max(result))
    CCs2 = np.array(CCs2)
    # Get second key_frames (second step)
    tmp = np.where(CCs2 >20)[0]
    index2 = np.insert(tmp + 1, 0, 0)
    key_frames_index = np.array([index[i] for i in index2])
    print(f"Final Key Frame Index: {key_frames_index}")
    
    final_key_frames = [imgs[i] for i in key_frames_index]
    # import pdb;pdb.set_trace()
    for f in final_key_frames:
        plt.imshow(f)
        plt.show()
    frame_count = 0
    # frame count + 几秒抽一帧
    for frame in final_key_frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cnts = detect_paragraph(gray)
        contours_text(gray, cnts, key_frames_index[frame_count])
        frame_count += 1



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



# if __name__ == "__main__":
#     main(sys.argv[1])
#     # index = [1,4,8,12,17,21,22,26,30,36,39,45,59,64,66]
#     # final_index = [11, 16, 35, 38]
#     # pathin = "/Users/yizhizhang/Downloads/test2.mp4"
#     # frames, duration, imgs = get_Canny(pathin)
#     # frames = [frames[i] for i in index]
#     # differentials = get_Differential(frames)
#     # CCs = []
#     # for differential in differentials:
#     #     #change to bool image
#     #     arr = np.asarray(differential)
#     #     arr = arr[0:int(0.23 * differential.shape[1]),:]
#     #     arr = arr != 255
#     #     # CC Analysis
#     #     result = connected_component_labelling(arr, 4)
#     #     CCs.append(np.max(result))
#     #     print(np.max(result))
#     # image = cv2.imread("/Users/yizhizhang/Downloads/test_frames/test3.png")
#     # #image = cv2.imread("/Users/yizhizhang/Desktop/andrew.png")
#     # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     # cnts = detect_paragraph(gray)
#     # contours_text(gray, cnts)
