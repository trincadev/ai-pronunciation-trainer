
import base64
import json
import tempfile

import soundfile as sf

from aip_trainer import app_logger
from aip_trainer.models.models import getTTSModel
from aip_trainer.models.AIModels import NeuralTTS


sampling_rate = 16000
model_de = getTTSModel('de')
model_TTS_lambda = NeuralTTS(model_de, sampling_rate)


def lambda_handler(event, context):

    body = json.loads(event['body'])

    text_string = body['value']

    linear_factor = 0.2
    audio = model_TTS_lambda.getAudioFromSentence(
        text_string).detach().numpy()*linear_factor
    with tempfile.TemporaryFile(prefix="temp_sound_", suffix=".wav") as f1:
        app_logger.info(f"Saving temp audio to {f1.name}...")
        # random_file_name = utilsFileIO.generateRandomString(20) + '.wav'
        # sf.write('./'+random_file_name, audio, 16000)

        sf.write(f1.name, audio, sampling_rate)
        with open(f1.name, "rb") as f:
            audio_byte_array = f.read()
        # os.remove(random_file_name)
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Headers': '*',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
            },
            'body': json.dumps(
                {
                    "wavBase64": str(base64.b64encode(audio_byte_array))[2:-1],
                },
            )
        }
