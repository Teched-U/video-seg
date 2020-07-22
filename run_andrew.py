import os
import click
import glob
from process_video import run



@click.command()
@click.option('-i', 'input_dir')
@click.option('-o', 'model_input_dir')
def main(input_dir:str, model_input_dir: str):
    video_files = glob.glob(f'{input_dir}/*.mp4')

    for video_file in video_files:
        video_name = os.path.basename(video_file)
        run(video_name, video_dir=input_dir, input_dir=model_input_dir)

if __name__ == '__main__':
    main()

