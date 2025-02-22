from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs import play

import os
import time

load_dotenv()

client = ElevenLabs(
    api_key=os.getenv("ELEVENLABS_API_KEY")
)

output_path = "output.mp3"

segments = [
    {"type": "text_to_sound_effects",
     "text": "sound of birds singing a beautiful song"},
    {"type": "pause",  "seconds": 1},
    {"type": "text_to_speech",
     "text": "Det var en fin sång"}
]

t0 = time.time()
with open(output_path, "wb") as f:
    for s in segments:
        t00 = time.time()
        print(f"Sending api call for {s['type']}"
        if s["type"] == "text_to_sound_effects":
            result = client.text_to_sound_effects.convert(
                text=s["text"],
            )

            for chunk in result:
                f.write(chunk)
        elif s["type"] == "pause":
            continue
        elif s["type"] == "text_to_speech":
            result = client.text_to_speech.convert(
                text=s["text"],
                voice_id="JBFqnCBsd6RMkjVDRZzb",
                model_id="eleven_multilingual_v2"
            )
        
            for chunk in result:
                f.write(chunk)
        t1 = time.time()
        print(f"Conversion time for {s['type']}: {t1 - t00} seconds")


t1 = time.time()
print(f"Conversion time: {t1 - t0} seconds")

print(f"Audio saved to {output_path}")