import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

audio_file = "sample_call.mp3"

# Step 1: Transcribe
print("Step 1: Transcribing audio...")
with open(audio_file, "rb") as file:
    transcription = client.audio.transcriptions.create(
        file=(audio_file, file.read()),
        model="whisper-large-v3",
    )
transcript = transcription.text
print("Transcription done.\n")

# Step 2: Analyze with Llama
print("Step 2: Analyzing the call...")

prompt = f"""You are an expert sales call analyst. Analyze the following sales call transcript between a sales POC and a business owner. Return ONLY a valid JSON object with these exact fields:

{{
  "summary": "A 2-3 sentence summary of the call",
  "customer_pain_points": ["list of pain points or requirements mentioned by the customer"],
  "objections_raised": ["list of objections or concerns raised by the customer"],
  "commitments_made": ["list of any commitments made by either party"],
  "next_action": "What should happen next, with suggested timeline",
  "sentiment": "positive / neutral / negative",
  "sentiment_reason": "One line explanation",
  "call_outcome": "interested / not interested / follow-up needed / closed",
  "quality_score": {{
    "introduction": {{"score": 0-10, "reason": "one line"}},
    "needs_understanding": {{"score": 0-10, "reason": "one line"}},
    "objection_handling": {{"score": 0-10, "reason": "one line"}},
    "clear_next_step": {{"score": 0-10, "reason": "one line"}},
    "professionalism": {{"score": 0-10, "reason": "one line"}},
    "overall_score": 0-10
  }}
}}

Transcript:
{transcript}

Return only the JSON, no other text."""

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": prompt}],
    response_format={"type": "json_object"},
)

analysis = json.loads(response.choices[0].message.content)

print("\n--- CALL ANALYSIS ---\n")
print(json.dumps(analysis, indent=2))