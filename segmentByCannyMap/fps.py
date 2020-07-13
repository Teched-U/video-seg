# #coding=utf-8
import cv2
# import numpy as np
import  os
# import time

def process_video_to_image(folder_path,  xvalue, yvalue, width, height):
    try:
        #通过视频位置读取视频
        vid_cap = cv2.VideoCapture(os.path.join(folder_path,"cropped.mp4"))
        
        #获取视频的总时长
        if vid_cap.isOpened():
            #获取视频的帧率
            rate=vid_cap.get(5)
            #获取视频的帧数
            FrameNumber=vid_cap.get(7)
            duration=FrameNumber/rate
            #视频的秒数
            print(duration)

        success, image = vid_cap.read()
        count = 0
        while success:
            temp = vid_cap.get(0)
            cv2.imwrite(folder_path + "/frames/" + str(count) + ".jpg", image)  # save frame as JPEG file
            count += 1
            vid_cap.set(cv2.CAP_PROP_POS_MSEC, 1 * 1000 * count)
            success, image = vid_cap.read()
            if temp == vid_cap.get(0):
                print("视频异常，结束循环")
                break
        print('Total frames: ', count)
    except:
        return False
    return True

if __name__ == "__main__":
    process_video_to_image("/Users/yizhizhang/Downloads/",200,200,200,200)
#200这几个参数是对图片的大小进行了设置

# '''
# for each vedio in vedios
#     分解视频成图像
# '''
# SourceImgPath = "/Users/yizhizhang/Downloads/cropped.mp4" # 视频读取路径
# #vedionamelist = os.listdir(SourceImgPath)  # 获得所有视频名字列表

# ImgWritePath = '/Users/yizhizhang/Downloads/frames/'  # 图像保存路径
# img_end = ".jpg"
# img_start = 0


# VedioPath = "/Users/yizhizhang/Downloads/cropped.mp4"# os.path.join('%s%s' % (SourceImgPath, vedio_path))  # 获得文件夹下所有文件的路径   读取路径和保存路径
# cap = cv2.VideoCapture(VedioPath)
# while cap.read():
#     # get a frame
#     start = time.time()
#     ret, frame = cap.read()
#     if ret == False:
#        break #读到文件末尾

#         # 显示第几帧
#     frames_num = cap.get(cv2.CAP_PROP_POS_FRAMES)
#         # 显示实时帧数
#     FPS = cap.get(cv2.CAP_PROP_FPS)
#         # 总帧数
#     total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
#         # show information of frame
#     cv2.putText(frame, "FPS:"+str(FPS), (17, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255))
#     cv2.putText(frame, "NUM OF FRAME:"+str(frames_num), (222, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255))
#     cv2.putText(frame, "TOTAL FRAME:" + str(total_frame), (504, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255))
#         # show a frame
#     #cv2.imshow("capture", frame)
#         # img name
#     img_name = str(img_start) + img_end
#     img_start = img_start + 1
#         # 存储
#     cv2.imwrite(ImgWritePath + img_name, frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     print(1.0 - time.time() + start)
#     time.sleep(1.0 - time.time() + start)

# cap.release()
# cv2.destroyAllWindows()
