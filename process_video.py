#!/usr/bin/env python

import json
import os 
import os.path as path
import sys
from preprocess import google_transcribe , combine_asr
from run_model import run_model
import tempfile
import cv2
import pickle
from pathlib import Path
import click


class ResultJson:

    def start():
        return {
            "state":  "处理字幕信息中",
            "done": False,
        }

    def asr_done(asr_result):
        return {
            "state":  "整合字幕信息中",
            "transcript": asr_result['all_transcript'],
            "done": False,
        }

    def shot_done(shot_result):
        return {
            "state": "初步片段分离完成",
            "done": False,
            "num_segs": shot_result["num_segs"],
            "avg_seg_dur": shot_result["avg_seg_dur"]
        }
    
    def story_done(story_list):
        return {
            "state": "短视频切割完成",
            "done": True,
            "story_list": story_list
        }

RESULT_DIR = '/data/results/'
VIDEO_DIR ='/data/upload/'
THUMBNAIL_DIR = '/data/thumbnails/'
FEATURE_DIR = '/data/features/'

class Story:
    def __init__(self):
        self.transcript = ""
        self.words = []
        self.timestamp = None
        self.duration = None
        self.thumbnail = None

    def merge_shot(self, shot):
        self.transcript += f" {shot['transcript']}"
        self.words += shot['words']
        if self.timestamp is None:
            self.timestamp = shot['timestamp']
            self.duration = shot['duration']
        else:
            self.duration = (shot['timestamp'] - self.timestamp) + shot['duration']

    def to_dict(self):
        return {
            'transcript': self.transcript,
            'timestamp': self.timestamp,
            'duration': self.duration,
            'words': self.words,
            'thumbnail': self.thumbnail
        }

    def extract_thumbnail(self, video, img_dir):
        video.set(cv2.CAP_PROP_POS_MSEC, 1000 * self.timestamp)
        success, image = video.read()
        if success:
            thumbnail_path = os.path.join(img_dir, f'{str(int(self.timestamp*1000))}.jpg')
            cv2.imwrite(thumbnail_path, image)
            self.thumbnail = thumbnail_path
        else:
            print(f"ERROR EXTRACTING {img_dir} at {self.timestamp}")
        


def aggregate_results(seg_result, input_file):
    with open(input_file, 'rb') as f:
        data = pickle.load(f)

    input_data = data['features'] 
    video_path = data['video_name']

    shot_buffer = []
    story_list = []
    for idx, (is_start, shot) in enumerate(zip(seg_result, input_data)):
        if idx == 0 or is_start:
            if shot_buffer:
                # Merge current story
                story = Story()
                for shot in shot_buffer:
                    story.merge_shot(shot)
                
                story_list.append(story)
                shot_buffer = []
            else:
                # First start a story
                shot_buffer.append(shot)
        else:
            # is within a story
            shot_buffer.append(shot)
    
    if shot_buffer:
        # Merge last story
        story = Story()
        for shot in shot_buffer:
            story.merge_shot(shot)
        story_list.append(story)


    # TODO(extract slide)
    video_name = Path(os.path.basename(video_path)).stem
    thumb_dir = os.path.join(THUMBNAIL_DIR, video_name)
    Path(thumb_dir).mkdir(parents=True, exist_ok=True)

    vid_cap = cv2.VideoCapture(video_path)
    if vid_cap.isOpened():
        for story in story_list:
            story.extract_thumbnail(vid_cap, thumb_dir)

    story_list = [story.to_dict() for story in story_list]

    return story_list
    

def write_result(video_path, data):
    print(f"Writing Data: {data} to {video_path}")
    video_name_no_prefix = Path(path.basename(video_path)).stem
    save_path = path.join(RESULT_DIR, f'{video_name_no_prefix}.json')

    with open(save_path, 'w') as f:
        json.dump(data, f)


@click.command()
@click.argument(
    'video_name'
)
@click.option(
    '--video-dir',
    "video_dir",
)
@click.option(
    '--asr-load',
    "asr_load_path",
    type=str,
    help='Absolute file path to LOAD the google ASR result'
)
@click.option(
    '--result-dir',
    'result_dir',
    type=str,
    help='Where results are saved in the end'
)
@click.option(
    '--asr-save',
    "asr_save_path",
    type=str,
    help='Absolute file path to SAVE the google ASR result '
)
def main(video_name, video_dir, result_dir, asr_load_path, asr_save_path):
    """
    Use this to segments a video.

    Example: \n
    Process '/data/videos/foo.mp4':\n
        `python process_video.py foo.mp4 --video-dir /data/videos`

    Process '/data/videos/foo.mp4' and save its Google ASR result:\n
        `python process_video.py foo.mp4 --video-dir /data/videos --asr_save /tmp/foo.data` 
    

    Process video and load ASR result:\n
        `python process_video.py foo.mp4 --video-dir /data/videos --asr_load /tmp/foo.data`

    Processed output will be in '/data/results' unless you override it with '--result-dir'

    Model features are stored in : '/data/features'

    """

    if video_dir:
        global VIDEO_DIR
        VIDEO_DIR = video_dir
    
    if result_dir:
        global RESULT_DIR
        RESULT_DIR = result_dir

    # Start the process
    write_result(video_name, ResultJson.start())

    # Send Google ASR
    video_path = path.join(VIDEO_DIR, video_name)
    if asr_load_path:
        with open(asr_load_path, 'rb') as f:
            result = pickle.load(f)
    else:
        result = google_transcribe([video_path], [asr_save_path])[0]

    write_result(video_name, ResultJson.asr_done(result))

    #  Combine ASR for Shots 
    fd, shot_outs = tempfile.mkstemp()
    result = combine_asr([result], [shot_outs])[0]
    write_result(video_name, ResultJson.shot_done(result))

    # Run Sequence Model  
    seg_result = run_model(video_path, shot_outs, FEATURE_DIR)

    # Aggregate Final results 
    story_list = aggregate_results(seg_result, shot_outs)    
    write_result(video_name, ResultJson.story_done(story_list))


if __name__ == '__main__':
    main()
