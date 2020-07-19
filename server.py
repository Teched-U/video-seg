from flask import Flask
from flask import request
from flask_socketio import SocketIO, emit, send
from flask_cors import CORS
from flask import g
import subprocess
import time


app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins = "*")

@app.route('/')
def hello_world():
        return 'NOT IMPLEMENTED'


def start_process(video_path):
        if 'requests' not in g:
                g.requests = {}
        
        cmd = f"python process_video {video_path}"
        proc = subprocess.Popen(cmd.split(), shell=True) 
        g.requests[video_path] = proc


        

@app.route('/upload', methods=['GET','POST'])
def upload():
        if request.method == 'POST':
                f = request.files['upload_file']
                save_path = '/data/upload/' + f.filename
                f.save(save_path)

                # Async start a routine
                start_process(save_path)

                return {'path': save_path}
        else:
                return 'GET OK' 

@app.route('/cancel/<int:video_id>')
def cancel(video_id):
        return f'Cancel {video_id}'


@socketio.on('message')
def handle_message(message):
    print('received message: ' + message)

@socketio.on('analysis')
def get_process_stats(arg):
    emit('timer', time.time())

if __name__ == '__main__':
        socketio.run(app)