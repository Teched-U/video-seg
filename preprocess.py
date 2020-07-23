#!/usr/bin/env python3
import click
import os
import os.path as path
import pickle
import glob
from typing import Tuple, List, Dict
import json
import cv2
from lib.ccl import *
import numpy as np

import wave

from pathlib import Path
from lib.extract_audio import extract_audio
from lib.vad import process
from lib.asr import transcribe, transcribe_ws
from lib.asr_google import sample_recognize, upload_blob
from lib.features import extract_features
import time


def get_canny(video: cv2.VideoCapture, cur_frame_ms: float):

    video.set(cv2.CAP_PROP_POS_MSEC, cur_frame_ms)
    success, frame = video.read()
    if not success:
        print(f"failed to read image at {cur_frame_ms}")
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 50, 150)

    return canny


def check_similarity(
    video: cv2.VideoCapture, cur_ts: float, cur_dur: float, next_ts: float, next_dur: float
) -> bool:

    # buffer window of 5 seconds 
    WINDOW_LEN = 3
    if cur_dur < WINDOW_LEN:
        print(f"shot duration only {cur_dur}")
        cur_frame_ms = cur_ts * 1000
    else:
        cur_frame_ms = (cur_ts + cur_dur - WINDOW_LEN) * 1000

    prev_canny = get_canny(video, cur_frame_ms)

    if next_dur < WINDOW_LEN:
        cur_frame_ms = (next_ts + next_dur) * 1000
    else:
        cur_frame_ms = (next_ts + WINDOW_LEN) * 1000

    next_canny = get_canny(video, cur_frame_ms)
    try:
        diff = next_canny - prev_canny
    except ValueError as e:
        click.secho(str(e), fg="red")
        return False

    # change to bool image
    arr = np.asarray(diff)
    arr = arr != 255
    # CC Analysis
    result = connected_component_labelling(arr, 4)

    result = np.max(result)

    # Almost same frame, so it's okay to just merge
    return result <= 20


def merge_segments(a: Dict, b: Dict) -> Dict:
    print(
        f"merging shots {a['timestamp']}({a['duration']}) <= {b['timestamp']}({b['duration']}"
    )
    a["duration"] = b["timestamp"] - a["timestamp"] + b["duration"]
    a["words"] += b["words"] 
    a["transcript"] = a["transcript"] + "\n" + b["transcript"]

    return a


def merge(video_file: str, segments: List[Dict]) -> List[Dict]:
    vid_cap = cv2.VideoCapture(video_file)

    if not vid_cap.isOpened():
        print(f"Failed to open video {video_file}")
        return []

    rate = vid_cap.get(cv2.CAP_PROP_FPS)
    frame_num = vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)

    cur_seg = segments[0]
    merged_segments = []
    # Start merging

    duration = [seg['duration'] for seg in segments]
    duration = np.array(duration)
    click.secho(f'[{video_file}]BEFORE: Duration - [mean: {np.mean(duration)}, max: {np.max(duration)}, min: {np.min(duration)}')
    click.secho(f'Number of segs: {len(segments)}')
    start = time.time()

    i  = 1
    while i < len(segments):
        cur_ts = cur_seg["timestamp"]
        next_seg = segments[i] 

        # If duration of previous segment already long
        while next_seg["timestamp"] - cur_ts <= 8:
            # A short segment, see if it has the identical visual content with the next one
            to_merge = check_similarity(
                vid_cap, cur_ts, cur_seg["duration"], next_seg["timestamp"], next_seg['duration']
            )
            
            if to_merge:
                cur_seg = merge_segments(cur_seg, next_seg)
            else:
                # don't merge then brea
                break
                    
            i+=1
            
            if i < len(segments):
                # check next seg
                next_seg = segments[i]
            else:
                # already last break
                break
        
        if cur_seg['duration'] < 5:
            click.secho(f"Duration={cur_seg['duration']}, i={i}, video: {video_file}", fg="red")
        merged_segments.append(cur_seg)
        cur_seg = next_seg

        if i == len(segments) -1:
            if cur_seg['duration'] < 5:
                click.secho(f"Duration={cur_seg['duration']}, i={i}, video: {video_file}", fg="red")
            # Last segment
            merged_segments.append(cur_seg)

        i+=1

    
    duration = [seg['duration'] for seg in merged_segments]
    duration = np.array(duration)
    click.secho(f'[{video_file}]BEFORE: Duration - [mean: {np.mean(duration)}, max: {np.max(duration)}, min: {np.min(duration)}')
    click.secho(f'Number of segs: {len(merged_segments)}')
    elapsed = time.time() - start
    click.secho(f'Time: {elapsed}')

    return merged_segments


def split_segment(video, ts, dur) -> List[Dict]:
    pass
    

def split_by_video(video_file: str, segments: List[Dict]) -> List[Dict]:
    vid_cap = cv2.VideoCapture(video_file)

    if not vid_cap.isOpened():
        print(f"Failed to open video {video_file}")
        return []

    rate = vid_cap.get(cv2.CAP_PROP_FPS)
    frame_num = vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)

    split_segs = []
    for seg in segments:
        # Split long segments
        if seg['duration'] > 45:
            segs = split_segment(vid_cap, seg['timestamp'], seg['duration'])
            split_segs += [seg]
        else:
            split_segs += [seg]

    return split_segs


def image_merge_preprocess(result_files: List[str], save_paths: str) -> List[Dict]:
    results = []
    for input_file, save_path in zip(result_files, save_paths):
        with open(input_file, 'rb') as fp:
            data = pickle.load(fp)

        # Try merge the segments
        video_path = data['video_name']
        segs = data['features']

        merged_segs = merge(video_path, segs)

        durations = [seg['duration'] for seg in merged_segs]
        durations = np.array(durations)

        with open(save_path, 'wb') as fp:
            data = {
                'video_name': video_path,
                'features': merged_segs 
            }
            pickle.dump(data, fp, pickle.HIGHEST_PROTOCOL)

        results.append({
            'num_segs': len(merged_segs),
            'avg_seg_dur': np.mean(durations)
        })

    return results


def process_video(video_file: str) -> Dict:

    audio_data = extract_audio(video_file)
    audio_file = video_file.split('.')[0] + ".wav"
    with open(audio_file, 'w+b') as fp:
        wf = wave.open(fp, "w")
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(np.frombuffer(audio_data, dtype=np.uint8))
        wf.close()

    uri = upload_blob("easytopic", audio_file, os.path.basename(audio_file))

    results = sample_recognize(uri)

    return results

    # segments = list(process(audio_data))

    # duration = [seg['duration'] for seg in segments]
    # duration = np.array(duration)
    # click.secho(f'BEFORE: Duration - [mean: {np.mean(duration)}, max: {np.max(duration)}, min: {np.min(duration)}')

    # segments = split_by_video(video_file, segments)

    # # segments = merge(video_file, segments)

    # # duration = [seg['duration'] for seg in segments]
    # # duration = np.array(duration)
    # # click.secho(f'AFTER: Duration - [mean: {np.mean(duration)}, max: {np.max(duration)}, min: {np.min(duration)}')

    # # Send each segment to Google Voice
    # feature_arr = []
    # previous_end_ts = 0.0
    # for segment in segments:
    #     transcript = "NONE"
    #     # try:
    #     #     transcript = sample_recognize(segment["bytes"], segment["duration"])
    #     # except Exception as e:
    #     #     click.secho(str(e), fg='red')

    #     #     click.secho(
    #     #         f"Translaton Failed on {segment['timestamp']}({segment['duration']}) from {video_file}",
    #     #         fg="red"
    #     #         )
    #     #     transcript = "Translation fails"
    #         

    #     # pitch, volume = extract_features(segment["bytes"])

    #     # pause_time = float(segment["timestamp"]) - previous_end_ts
    #     feature = {
    #     #     "pause": pause_time,
    #     #     "pitch": pitch,
    #     #     "volume": volume,
    #     #     "transcript": transcript,
    #         "timestamp": segment["timestamp"],
    #         "duration": segment["duration"],
    #     }
    #     # previous_end_ts = float(segment["timestamp"]) + float(segment["duration"])
    #     feature_arr.append(feature)

    # return {"features": feature_arr, "video_name": video_file}
def google_transcribe(video_paths, output_paths):
    """
    Input: 
        video_paths: List[path],
        output_paths: List[path], 
            Optional files to store the result (to prevent multiple runs).
            Use None to indicate not saving
    Return:
        data_list: List[Dict]
    """
    data_list = []
    for video_path, output_path in zip(video_paths, output_paths):
        data = {}
        click.secho(f"processing {video_path}", fg="red")
        try:
            data = process_video(video_path)
            
        except Exception as e:
            print(f"Error: {e}")
            continue
        
        all_transcript = [alter.transcript for alter in data]
        all_transcript = '\n'.join(all_transcript)

        result = {
            "results": data,
            "video_name": video_path,
            "all_transcript": all_transcript
        }
        data_list.append(result)

        # Output to a json file if chosen
        if output_path:
            with open(output_path, 'wb') as f:
                pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)


    return data_list


def combine_asr(data_list: List[Dict], output_paths: List[str]) -> List[Dict]:
    """
    input: [{
        "transcript": ...,
        "words": [
            {
                word: ..., 
                start_time: 
                    seconds: ...
                    nanos: ...
                end_time: 
                    seconds: ...
                    nanos: ...
            }
        ]
    }]

    Output: 
    {
        features: [{
                    "transcript": ...,
                    "timestamp": ...,
                    "duration": ...,
                    "words": [Word],
                }],
        video_name: <path to video>
    }
    """
    log_result_list = []
    for data, output_path in zip(data_list, output_paths):
        result = data["results"]
        video_path = data["video_name"]

        segments = []
        for alternative in result:
            segment = {}
            ts = [w.start_time.seconds + w.start_time.nanos / 1000000000 for w in alternative.words]
            ts = sorted(ts)
            segment['timestamp'] = ts[0]
            segment['duration'] = ts[-1] - ts[0]
            segment['transcript'] = alternative.transcript
            segment['words'] = [
                {
                    "word": w.word,
                    "start_time": w.start_time.seconds + w.start_time.nanos/1000000000,
                    "end_time": w.end_time.seconds + w.end_time.nanos/1000000000,
                }
                for  w in alternative.words
            ]
            segments.append(segment)
        
        # Save it 
        with open(output_path, 'wb') as f:
            data = {
                "features": segments, 
                "video_name": video_path
            }
            pickle.dump(data,f, pickle.HIGHEST_PROTOCOL)

        # Output for log result
        duration_list = [seg['duration'] for seg in segments]
        avg_seg_dur = np.mean(np.array(duration_list))
        log_result = {
            "num_segs": len(segments),
            "avg_seg_dur": avg_seg_dur
        }
        log_result_list.append(log_result)

    return log_result_list


def process_video_easy(video_file: str, version: str) -> Dict:
    audio_data = extract_audio(video_file)

    if version == "v1":
        segments = process(audio_data)

        feature_arr = []
        previous_end_ts = 0.0
        for idx, segment in enumerate(segments):
            # Get transcript
            result = transcribe(segment["bytes"])
            transcript = result["hypotheses"][0]["utterance"]
            print(transcript)

            # Get features
            pitch, volume = extract_features(segment["bytes"])
            pause_time = float(segment["timestamp"]) - previous_end_ts
            feature = {
                "pause": pause_time,
                "pitch": pitch,
                "volume": volume,
                "transcript": transcript,
                "timestamp": segment["timestamp"],
                "duration": segment["duration"],
            }
            previous_end_ts = float(segment["timestamp"]) + float(segment["duration"])
            feature_arr.append(feature)

        return {"features": feature_arr, "video_name": video_file}
    else:
        result = transcribe_ws(audio_data)
        feature_arr = [
            {"transcript": data["transcript"], "timestamp": data["segment-start"]}
            for data in result
        ]
        print("Sleeping for a while for the worker to get back")
        time.sleep(15)
        return {"features": feature_arr, "video_name": video_file}
    



@click.command()
@click.option("-i", "--input-path", type=str, help="ABSOLUTE path to video folders")
@click.option("-s", "source", default="easytopic", type=str, help="Data souce")
@click.option("-o", "output_dir", type=str, help="Where to save the transcripts")
@click.option("--cmd", "cmd", type=click.Choice(["asr_google", "post_proc", "videos", "coursera", "merge"]), help="Command")
@click.option("-v", "videos", multiple=True, help="videos to process")
def main(input_path: str, source: str, output_dir: str, cmd: str, videos:List[str]) -> None:
    if cmd == "asr_google":
        videos = next(os.walk(input_path))[1]
        video_paths = [ 
            os.path.join(input_path, video_folder, f"{video_folder}.mp4")
            for video_folder in videos
        ]
        google_transcribe(videos, output_dir)
    elif cmd == "post_proc":
        output_files = next(os.walk(output_dir))[2]
        output_files = [os.path.join(output_dir, p) for p in output_files]
        combine_asr(output_files)
    elif cmd == "videos":
        google_transcribe(videos, output_dir)
        output_files = next(os.walk(output_dir))[2]
        output_files = [os.path.join(output_dir, p) for p in output_files]
        combine_asr(output_files)
    elif cmd == 'coursera':
        videos = glob.glob(f"{input_path}/*.mp4")
        google_transcribe(videos, output_dir)
        output_files = next(os.walk(output_dir))[2]
        output_files = [os.path.join(output_dir, p) for p in output_files]
        combine_asr(output_files)
    elif cmd == "merge":
        result_dir = input_path
        result_files = videos
        if result_dir:
            result_files = glob.glob(f"{result_dir}/*.json")
            print(result_files)

        save_paths = []
        for input_file in result_files:
            file_name = os.path.basename(input_file)
            save_path = os.path.join(output_dir, file_name)
            save_paths.append(save_path)

        image_merge_preprocess(result_files, save_paths)
        




    # if source == "easytopic":
    #     videos = next(os.walk(input_path))[1]
    #     for video_folder in  videos:
    #         save_path = os.path.join(output_dir, f"{video_folder}.json")
    #         if path.exists(save_path): print(f"{save_path} exists. Skip")
    #             continue
    #         with open(save_path, 'wb') as f:
    #             data = {}
    #             try:
    #                 data = process_video(
    #                     os.path.join(input_path, video_folder, f"{video_folder}.mp4"), version)
    #             except Exception as e:
    #                 click.secho(f"Error: {e}")
    #                 continue
    #
    #             pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    # else:
    #     raise ValueError(f"not supported {source}")


if __name__ == "__main__":
    main()
