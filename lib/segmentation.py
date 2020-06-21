#!/usr/bin/env python3
from nltk import tokenize, pos_tag
import matplotlib.pyplot as plt
import numpy as np
from statsmodels import robust
import seaborn as sns

import nltk

# FIXME: why am I stuck at below
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


'''Shot representation'''
class Shot:
    def __init__(self, id, pitch, volume, pause, mfcc_vector, init_time, end_time):
        self.id = id            #shot id
        self.pitch = pitch      #pitch value
        self.volume = volume    #volume contained in a chunk
        self.pause_duration  = pause # pause time before the shot being voiced
        self.surprise = 0  #bayesian surprise value of f0 in a windowed audio signal
        self.transcript = None  #transcription from ASR of a shot
        self.ocr = None #text extracted from ocr
        self.mfcc_vector = mfcc_vector
        self.adv_count = 0
        self.init_time = init_time
        self.end_time = end_time
        self.duration = end_time - init_time
        self.word2vec = None
        self.valid_vector = None

    '''extract the transcripts and related concepts from CSO ontology'''
    def extractTranscriptAndConcepts(self, transcript, ocr_on, docSim):

        aux = ""
        #f2 = open(video_path + "transcript/transcript"+str(self.id)+".txt")
        a = transcript
        cue_phrases = ['actually',  'further',  'otherwise', 'also' , 'furthermore'
         'right' , 'although',  'generally',
        'say',  'and', 'however',  'second', 'basically',  'indeed',  'see',
        'because', 'let' ,'similarly','but','look', 'since',
        'essentially', 'next', 'so',
        'except', 'no' ,'then',
        'finally' ,'now', 'therefore',
        'first', 'ok', 'well',
        'firstly', 'or', 'yes' ]

        words = tokenize.word_tokenize(a, language='english')

        words = [word.lower() for word in words]
        if words:
            if words[0] in cue_phrases:
                self.adv_count = 1
            else:
                self.adv_count = 0
        else:
            self.adv_count = 0

        '''Apply pos_tag in the transcript to extract only adjectives and nouns'''
        words = [word for (word, pos) in pos_tag(words) if pos == 'NN' or
        pos == 'JJ' or pos == 'NNS' or pos == 'JJR' or pos == 'JJR']

        transcript = ' '.join(words)


        self.transcript = transcript
        self.valid_vector, self.word2vec = docSim.vectorize(self.transcript)
        #f2.close()




class DocSim(object):
    def __init__(self, w2v_model , stopwords=[]):
        self.w2v_model = w2v_model
        self.stopwords = stopwords

    def vectorize(self, doc):
        """Identify the vector values for each word in the given document"""
        doc = doc.lower()
        words = [w for w in doc.split(" ") if w not in self.stopwords]
        word_vecs = []
        for word in words:
            try:
                vec = self.w2v_model[word]
                word_vecs.append(vec)
            except KeyError:
                # Ignore, if the word doesn't exist in the vocabulary
                pass

        # Assuming that document vector is the mean of all the word vectors
        # PS: There are other & better ways to do it.
        if not word_vecs:
            return False ,np.zeros((300,))

        vector = np.mean(word_vecs, axis=0)

        return True, vector


    def _cosine_sim(self, vecA, vecB):
        """Find the cosine similarity distance between two vectors."""
        csim = np.dot(vecA, vecB) / (np.linalg.norm(vecA) * np.linalg.norm(vecB))
        if np.isnan(np.sum(csim)):
            return 0
        return csim

    def calculate_similarity(self, source_doc, target_docs=[], threshold=0):
        """Calculates & returns similarity scores between given source document & all
        the target documents."""
        if isinstance(target_docs, str):
            target_docs = [target_docs]

        not_empty ,source_vec = self.vectorize(source_doc)
        results = []
        for doc in target_docs:
            not_empty, target_vec = self.vectorize(doc)
            sim_score = self._cosine_sim(source_vec, target_vec)

            results.append({
                'score' : sim_score,
                'doc' : doc
            })
            # Sort results by score in desc order
            #results.sort(key=lambda k : k['score'] , reverse=True)


        return results
