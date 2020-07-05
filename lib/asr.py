#!/usr/bin/env python3

import http.client
import os
import tempfile
import wave
import numpy as np
import urllib
import json
from concurrent.futures import ThreadPoolExecutor
import time

from .asr_client import MyClient


def transcribeAudio(path_to_audio_file, samplerate=16000):
    headers = {"Content-type": "audio/wav; codec=\"audio/pcm\"; samplerate=" + str(samplerate), "Transfer-Encoding": "chunked"}

    with open(path_to_audio_file, 'rb') as audio_file:
        response = ""
        try:

            body = audio_file.read()
            # Connect to server to recognize the wave binary

            host = 'localhost'
            port = 8080
            # host = os.environ['ASR_SERVER']
            # port = int(os.environ['GSTREAM_PORT'])

            conn = http.client.HTTPConnection(host=host, port=port)

            conn.request("POST", "/client/dynamic/recognize",
                         body, headers)
            response = conn.getresponse().read().decode("UTF-8")
            conn.close()
        except Exception as e:
            print(e)
        finally:
            audio_file.close()

        return response

def transcribe(audio_chunk):
    with tempfile.NamedTemporaryFile(mode='w+b', suffix='.wav') as fp:
        succes = False
        while not succes:
            try:
                sample_rate = 16000
                wf = wave.open(fp, 'w')
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(np.frombuffer(audio_chunk, dtype=np.uint8))
                wf.close()
                res = transcribeAudio(fp.name)
                #print(res, flush=True)
                res = json.loads(res)
                if res:
                    return res
                    succes = True
            except Exception as e:
                print(e)
                print('trying again', flush=True)
                time.sleep(0.5)


def transcribe_ws(
    audio_chunk, 
    save_adaptation_state = None, 
    send_adaptation_state = None,
    uri :str = "ws://localhost:8080/client/ws/speech",
    content_type :str = "",
    ):
    fd, path = tempfile.mkstemp(suffix='.wav')
    with open(path, 'w+b') as fp:
        sample_rate = 16000
        wf = wave.open(fp, 'w')
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(np.frombuffer(audio_chunk, dtype=np.uint8))
        wf.close()
    with open(path, 'r+b') as fp:
        try:
            ws = MyClient(
                fp, 
                uri + f'?{urllib.parse.urlencode([("content-type", content_type)])}',
                ThreadPoolExecutor(),
                byterate=sample_rate * 2,
                save_adaptation_state_filename=save_adaptation_state, 
                send_adaptation_state_filename=send_adaptation_state
                )
        except Exception as e:
            print(str)
        finally:
            result = ws.get_result()
            return result
        


