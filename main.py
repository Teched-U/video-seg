#!/usr/bin/env python3

import glob
import os.path as path
from typing import Tuple, List

from lib.extract_audio import extract_audio
from lib.vad import process
from lib.asr import transcribe
from lib.features import extract_features
from lib.segmentation import Shot, DocSim
from lib.genetic_algo import GeneticAlgorithm as GA
from gensim.models.keyedvectors import KeyedVectors


def init_word2vec(model_path:str, stopwords_file:str) -> Tuple[DocSim, List[str]] :
    with open(stopwords_file, 'r') as f:
        stopwords = f.read().split(",")
        model = KeyedVectors.load_word2vec_format(model_path, binary=True, limit=1000000)
        docSim = DocSim(model, stopwords=stopwords)

        return docSim, stopwords
    


def process_video(video_file: str, docSim: DocSim, stopwords: List[str]) -> None:
    audio_data = extract_audio(video_file)
    segments = process(audio_data)

    feature_arr = []
    transcript_arr = []
    chunks = []
    previous_end_ts = 0.0
    for idx, segment in enumerate(segments):
        # Get transcript
        result = transcribe(segment['bytes'])
        transcript = result['hypotheses'][0]['utterance']
        transcript_arr.append(transcript)

        # Get features
        pitch, volume = extract_features(segment['bytes'])
        pause_time = float(segment['timestamp']) - previous_end_ts
        feature = {
            'init_time': segment['timestamp'],
            'pause': pause_time,
            'pitch': pitch,
            'volume': volume 
        }
        previous_end_ts = float(segment['timestamp']) + float(segment['duration'])
        feature_arr.append(feature)

        # Create shot
        shot = Shot(
            idx, pitch, volume, 
            pause_time, [], 
            init_time=segment['timestamp'],end_time=0
        )
        shot.extractTranscriptAndConcepts(transcript, False, docSim=docSim)
        chunks.append(shot)

    chunks = [s for s in chunks if s.valid_vector]
    if len(chunks) < 2:
        boundaries = [0]
    else:
        '''calls the genetic algorithm'''
        ga = GA(population_size=100, constructiveHeuristic_percent=0.3, mutation_rate=0.05,
                                 cross_over_rate=0.4, docSim=docSim, shots=chunks,
                                 n_chunks=len(chunks), generations=500, local_search_percent=0.3,
                                 video_length=100, stopwords=stopwords, ocr_on=False)
        boundaries = ga.run()
    
    print(boundaries, flush=True)
    




if __name__ == '__main__':

    video_files = glob.glob('data/*.mp4')
    GOOGLE_MODEL_PATH = '/media/word2vec/GoogleNews-vectors-negative300.bin'
    STOPWORD_PATH = 'data/stopwords_en.txt'

    docsim_model, stopwords = init_word2vec(GOOGLE_MODEL_PATH, STOPWORD_PATH)

    for video_file in video_files:
        data = process_video(path.abspath(video_file), docsim_model, stopwords)
