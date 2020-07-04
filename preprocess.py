#!/usr/bin/env python3
import click
import os
import os.path as path
import pickle
from typing import Tuple, List, Dict

from lib.extract_audio import extract_audio
from lib.vad import process
from lib.asr import transcribe
from lib.features import extract_features


def process_video(video_file: str) -> Dict:
    audio_data = extract_audio(video_file)
    segments = process(audio_data)

    feature_arr = []
    previous_end_ts = 0.0
    for idx, segment in enumerate(segments):
        # Get transcript
        result = transcribe(segment['bytes'])
        transcript = result['hypotheses'][0]['utterance']
        print(transcript)

        # Get features
        pitch, volume = extract_features(segment['bytes'])
        pause_time = float(segment['timestamp']) - previous_end_ts
        feature = {
            'pause': pause_time,
            'pitch': pitch,
            'volume': volume,
            "transcript": transcript,
            "timestamp": segment['timestamp'],
            'duration': segment['duration'],
        }
        previous_end_ts = float(segment['timestamp']) + float(segment['duration'])
        feature_arr.append(feature)
    
    return {
        "features": feature_arr,
        "video_name": video_file
    }

@click.command()
@click.option(
    "-i",
    "--input-path",
    type=str,
    help="ABSOLUTE path to video folders"
)
@click.option(
    "-s",
    "source",
    default="easytopic",
    type=str,
    help="Data souce"
)
@click.option(
    "-o",
    "output_dir",
    type=str,
    help="Where to save the transcripts"
)
def main(
    input_path: str,
    source: str,
    output_dir : str
    ) -> None:

    if source == "easytopic":
        videos = next(os.walk(input_path))[1]
        for video_folder in  videos:
            save_path = os.path.join(output_dir, f"{video_folder}.json")
            if path.exists(save_path):
                print(f"{save_path} exists. Skip")
                continue
            with open(save_path, 'wb') as f:
                data = {}
                try:
                    data = process_video(
                        os.path.join(input_path, video_folder, f"{video_folder}.mp4"))
                except Exception as e:
                    click.secho(f"Error: {e}")
                    continue
                
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    else:
        raise ValueError(f"not supported {source}")


if __name__ == '__main__':
    main()