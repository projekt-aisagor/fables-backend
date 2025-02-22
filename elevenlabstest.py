from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs import play
import os
import time
load_dotenv()
client = ElevenLabs(
    api_key=os.getenv("ELEVENLABS_API_KEY")
)
t0 = time.time()
print("sending api call")
audio = client.text_to_speech.convert(
    text="I prinsessans värld vaknar prinsessan en vacker morgon. Hon går ner till köket för att bjuda sina vänner på tårta, som hon alltid gör när äventyren är klara.",
    voice_id="JBFqnCBsd6RMkjVDRZzb",
    model_id="eleven_multilingual_v2",
    output_format="mp3_44100_128",
)
print("api call took:")
print(time.time() - t0)
play(audio)