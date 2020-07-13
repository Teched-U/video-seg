import numpy as np
import cv2
import time
from ccl import *
from matplotlib import pyplot as plt


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







if __name__ == "__main__": main()