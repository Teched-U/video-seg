#!/usr/bin/env python3

import tempfile
import subprocess
from scipy.io import wavfile
import numpy as np


def extract_audio(file):
    with tempfile.NamedTemporaryFile(mode='w+b', suffix='.wav') as fp_a:
        command = "ffmpeg -y -i " + file + " -ab 160k -ac 1 -ar 16000 -vn " + fp_a.name

        subprocess.call(command, shell=True)

        fs, data = wavfile.read(fp_a.name)
        print(len(bytes(data)))
        
        return bytes(data)







