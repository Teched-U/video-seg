from google.cloud import speech_v1
from google.cloud.speech_v1 import enums
import io
import tempfile
import wave
import numpy as np


from google.cloud import storage


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)
    uri = f"gs://{bucket_name}/{destination_blob_name}"

    return uri



def sample_recognize(storage_uri):
    """
    Transcribe a short audio file using synchronous speech recognition
    """

    client = speech_v1.SpeechClient()


    # The language of the supplied audio
    language_code = "en-US"

    # Sample rate in Hertz of the audio data sent
    sample_rate_hertz = 16000

    # Encoding of audio data sent. This sample sets this explicitly.
    # This field is optional for FLAC and WAV audio formats.
    encoding = enums.RecognitionConfig.AudioEncoding.LINEAR16
    config = {
        "language_code": language_code,
        "sample_rate_hertz": sample_rate_hertz,
        "enable_automatic_punctuation": True, 
        "enable_word_time_offsets": True,
        "encoding": encoding,
    }

    audio = {"uri": storage_uri}

    operation = client.long_running_recognize(config, audio)

    print(u"Waiting for operation to complete...")
    response = operation.result()

    return_obj = []
    for result in response.results:
        # First alternative is the most probable result
        alternative = result.alternatives[0]
        print(u"Transcript: {}".format(alternative.transcript))
        return_obj.append(alternative)

    return return_obj