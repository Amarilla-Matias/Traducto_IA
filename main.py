import gradio as gr
import whisper
from translate import Translator
from dotenv import dotenv_values
from elevenlabs.client import ElevenLabs 
from elevenlabs import VoiceSettings


config = dotenv_values(".env")
ELEVENLABS_API_KEY = config["ELEVENLABS_API_KEY"]

def traductor(audio_file):
    try:
        model = whisper.load_model("base")
        result = model.transcribe(audio_file, fp32=True)
        transcription = result["text"]
    except Exception as e:
        raise gr.Error(
            f"Se ha producido un error transcribiendo el texto: {str(e)}")
        

    try:
        en_transcripcion = Translator(to_lang="en").translate(transcription)
    except Exception as e:
        raise gr.Error(
            f"Se ha producido un error traduciendo el texto:{str(e)}"
        )

    try:
        client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

        responce = client.text_to_speech.convert(
            voice_id="bIHbv24MWmeRgasZH58o",
            optimize_streaming_latency="0",
            output_format="mp3_22050_32",
            text= en_transcripcion,
            model_id="eleven_turbo_v2",
            voice_settings=VoiceSettings(
                stability=0.0,
                similarity_boost=1.0,
                style= 0.0,
                use_speaker_boost=True,
            ),
        )

        save_file_path = "audios/en.mp3"

        with open(save_file_path, 'wb') as f:
         for chunk in responce:
             if chunk:
                f.write(chunk)

        return save_file_path

    except Exception as e:
        raise gr.Error(
            f"Se ha producido un error creando el Audio:{str(e)}"
        )



web = gr.Interface(
    fn=traductor,
    inputs= gr.Audio(
        sources=['microphone'],
        type='filepath',
        label="Español"
    ),
    outputs=[gr.Audio(label="Ingles")],
    title='Traductor de Voz',
    description='Traductor de voz con IA a varios idiomas'
)

web.launch()