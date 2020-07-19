import json
import os 
import sys
from preprocess import google_transcribe , combine_asr
import tempfile


class ResultJson:

    def start():
        return {
            "state":  "处理字幕信息中"
        }

    def asr_done(asr_result):
        return {
            "state":  "整合字幕信息中",
            "transcript": asr_result['all_transcript'],
        }

    def shot_done(shot_result):
        return {
            "state": "初步片段分离完成",
            "num_segs": shot_result["num_segs"],
            "avg_seg_dur": shot_result["avg_seg_dur"]
        }

RESULT_DIR = '/data/results/'

def write_result(video_path, data):

    save_path = path.join(RESULT_DIR, video_path)

    with open(save_path, 'w') as f:
        f.write(json.dump(data))


if __name__ == '__main__':

    video_path = sys.argv[1]

    # Start the process
    write_result(video_path, ResultJson.start())

    # Send Google ASR
    fd, gout = tempfile.mkstemp()
    result = google_transcribe([video_path], [gout])
    write_result(video_path, ResultJson.asr_done(result))

    #  Combine ASR for Shots 
    fd, shot_outs = tempfile.mkstemp()
    result = combine_asr([gout], [shot_outs])
    write_result(video_path, ResultJson.shot_done(result))


    # Run Sequence Model  


