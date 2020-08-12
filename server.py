from flask import Flask
from flask import request
from flask_socketio import SocketIO, emit, send
from flask_cors import CORS
from flask import g
import subprocess
import time
import json
import os

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
        
        # print(f"here {cmd}")
        # Wait here 
        # try:
        #         outs, errs = proc.communicate(timeout=5)
        # except subprocess.TimeoutExpired:
        #         proc.kill()
        #         outs, errs = proc.communicate()
        # print(outs)
        # print(errs)



        g.requests[video_path] = proc


        

@app.route('/upload', methods=['GET','POST'])
def upload():
        if request.method == 'POST':
                f = request.files['upload_file']
                save_path = '/data/upload/' + f.filename
                f.save(save_path)
                # Async start a routine
                start_process(f.filename)

                return {'path': f.filename}
        else:
                return 'GET OK' 

@app.route('/cancel/<int:video_id>')
def cancel(video_id):
        return f'Cancel {video_id}'


@socketio.on('message')
def handle_message(message):
    print('received message: ' + message)

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