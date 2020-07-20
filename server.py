from flask import Flask
from flask import request
from flask_socketio import SocketIO, emit, send
from flask_cors import CORS
from flask import g
import subprocess
import time
import json
import os
from lib.asr_google import upload_blob

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins = "*")
RESULT_DIR='/data/results'

@app.route('/')
def hello_world():
        return 'NOT IMPLEMENTED'


def start_process(video_path):
        if 'requests' not in g:
                g.requests = {}
        
        cmd = f"python process_video.py {video_path}"
        proc = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE) 
        g.requests[video_path] = proc


CDN_URL = 'http://35.244.161.66/'

@app.route('/upload', methods=['GET','POST'])
def upload():
        if request.method == 'POST':
                f = request.files['upload_file']
                save_path = '/data/upload/' + f.filename
                f.save(save_path)
                # Async start a routine
                start_process(f.filename)

                # Upload to GC
                upload_blob('techedu-video-upload', save_path, f.filename)

                return {'path': f.filename, 'video_url': CDN_URL+f.filename}
        else:
                return 'GET OK' 

@app.route('/cancel/<int:video_id>')
def cancel(video_id):
        return f'Cancel {video_id}'


@socketio.on('analysis')
def get_process_stats(video_name):
        # Query the json log
        video_name_no_suffix = os.path.basename(video_name).split('.')[0]
        result_path = os.path.join(RESULT_DIR, f'{video_name_no_suffix}.json')
        if os.path.exists(result_path):
                with open(result_path, 'r') as fp:
                        result = json.load(fp)
                        print(result)
                        emit('result', result)
        else:
                emit('result', 'NO ready yet')



if __name__ == '__main__':
        socketio.run(app)