import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

audio_file = "sample_call.mp3"

print("Transcribing... please wait.")

with open(audio_file, "rb") as file:
    transcription = client.audio.transcriptions.create(
        file=(audio_file, file.read()),
        model="whisper-large-v3",
    )

print("\n--- TRANSCRIPT ---\n")
print(transcription.text)