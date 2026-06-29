import os
import io
import json
import zipfile
import subprocess
import requests
import html as html_lib
from datetime import datetime, timezone
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from mutagen import File as MutagenFile

load_dotenv()

# Groq client - used for Whisper transcription + LLaMA analysis
groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])

FS_DOMAIN = st.secrets["FRESHSALES_DOMAIN"]
FS_API_KEY = st.secrets["FRESHSALES_API_KEY"]

FS_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "Authorization": f"Token token={FS_API_KEY}"
}

AUDIO_EXTENSIONS = (".mp3", ".wav", ".m4a", ".ogg", ".flac", ".mpeg", ".mpg")


# ===== Freshsales helpers =====

def find_contact_by_phone(phone):
    url = f"https://{FS_DOMAIN}/crm/sales/api/search"
    params = {"q": phone, "include": "contact"}
    r = requests.get(url, headers=FS_HEADERS, params=params)
    if r.status_code != 200:
        return None
    results = r.json()
    for item in results:
        if item.get("type") == "contact":
            return item
    return None


def find_routing_target(contact_id):
    url = f"https://{FS_DOMAIN}/crm/sales/api/contacts/{contact_id}?include=deals,sales_account"
    r = requests.get(url, headers=FS_HEADERS)
    if r.status_code != 200:
        return ("contact", contact_id, None)
    data = r.json()
    deals = data.get("deals", [])
    sas = data.get("sales_accounts", [])
    if deals:
        return ("deal", deals[0]["id"], deals[0].get("name"))
    if sas:
        return ("sales_account", sas[0]["id"], sas[0].get("name"))
    return ("contact", contact_id, None)


def push_phone_call(target_type, target_id, phone):
    payload = {
        "phone_call": {
            "call_direction": True,
            "targetable_type": target_type,
            "targetable": {"id": target_id, "phone": phone}
        }
    }
    r = requests.post(f"https://{FS_DOMAIN}/crm/sales/api/phone_calls",
                      headers=FS_HEADERS, json=payload)
    return r.status_code in (200, 201)


def push_note(target_type, target_id, html_body):
    type_map = {"deal": "Deal", "sales_account": "SalesAccount", "contact": "Contact"}
    payload = {
        "note": {
            "description": html_body,
            "targetable_type": type_map.get(target_type, target_type),
            "targetable_id": target_id,
        }
    }
    r = requests.post(f"https://{FS_DOMAIN}/crm/sales/api/notes",
                      headers=FS_HEADERS, json=payload)
    return r.status_code in (200, 201)


def build_note_html(analysis, contact_name, duration_seconds):
    def esc(s):
        return html_lib.escape(str(s)) if s is not None else "Not mentioned"

    def list_html(items):
        if not items:
            return "None"
        return "<br>".join("- " + esc(i) for i in items)

    villa = analysis.get("villa_details", {})
    owner = analysis.get("owner_profile", {})
    comm = analysis.get("commercials_discussed", {})

    parts = []
    parts.append("<b>AI CALL ANALYSIS - " + esc(contact_name) + "</b><br>")
    parts.append("<b>Call Stage:</b> " + esc(analysis.get('call_stage')) + " | <b>Outcome:</b> " + esc(analysis.get('call_outcome')) + " | <b>Duration:</b> " + str(duration_seconds//60) + "m " + str(duration_seconds%60) + "s<br><br>")
    parts.append("<b>SUMMARY</b><br>" + esc(analysis.get('summary')) + "<br><br>")
    parts.append("<b>VILLA DETAILS</b><br>")
    parts.append("- Location: " + esc(villa.get('location')) + "<br>")
    parts.append("- Bedrooms: " + esc(villa.get('bedrooms')) + "<br>")
    parts.append("- Status: " + esc(villa.get('property_status')) + "<br>")
    amenities = villa.get('amenities', [])
    parts.append("- Amenities: " + (esc(', '.join(amenities)) if amenities else "Not mentioned") + "<br>")
    if villa.get('any_other_notable_features'):
        parts.append("- Notable: " + esc(villa.get('any_other_notable_features')) + "<br>")
    parts.append("<br>")
    parts.append("<b>OWNER PROFILE</b><br>")
    parts.append("- Current usage: " + esc(owner.get('current_usage')) + "<br>")
    parts.append("- Existing tie-ups: " + esc(owner.get('existing_tie_ups')) + "<br>")
    parts.append("- Decision maker: " + esc(owner.get('decision_maker')) + "<br>")
    parts.append("- Interest level: " + esc(owner.get('interest_level')) + "<br><br>")
    parts.append("<b>OWNER CONCERNS</b><br>" + list_html(analysis.get('owner_concerns', [])) + "<br><br>")
    parts.append("<b>OWNER REQUIREMENTS</b><br>" + list_html(analysis.get('owner_requirements', [])) + "<br><br>")
    parts.append("<b>COMMERCIALS DISCUSSED</b><br>")
    parts.append("- Owner expectation: " + esc(comm.get('revenue_expectation')) + "<br>")
    parts.append("- Model offered: " + esc(comm.get('revenue_share_or_model_mentioned')) + "<br>")
    parts.append("- Contract terms: " + esc(comm.get('contract_or_exclusivity')) + "<br><br>")
    parts.append("<b>COMMITMENTS MADE</b><br>" + list_html(analysis.get('commitments_made', [])) + "<br><br>")
    parts.append("<b>NEXT ACTION</b><br>" + esc(analysis.get('next_action')) + "<br><br>")
    parts.append("- Generated by AI Call Analyzer")
    return "".join(parts)


# ===== Audio helpers =====

def compress_audio(input_path, output_path):
    """Compress audio to mono 16kHz 64k mp3 via ffmpeg."""
    subprocess.run([
        "ffmpeg", "-y", "-i", input_path,
        "-ac", "1", "-ar", "16000", "-b:a", "64k",
        output_path
    ], check=True, capture_output=True)


def get_duration(path):
    try:
        audio = MutagenFile(path)
        return int(audio.info.length) if audio and audio.info else 60
    except Exception:
        return 60


def transcribe(path):
    with open(path, "rb") as f:
        transcription = groq_client.audio.translations.create(
            file=("audio.mp3", f.read()),
            model="whisper-large-v3",
        )
    return transcription.text


def analyze(transcript):
    prompt = ANALYSIS_PROMPT_TEMPLATE.replace("__TRANSCRIPT__", transcript)
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(response.choices[0].message.content)


def phone_from_filename(filename):
    """Extract phone number from filename e.g. '+917006347128.mp3' -> '+917006347128'"""
    name = os.path.splitext(os.path.basename(filename))[0]
    # Keep only digits and leading +
    cleaned = ""
    for i, ch in enumerate(name):
        if ch == "+" and i == 0:
            cleaned += ch
        elif ch.isdigit():
            cleaned += ch
    return cleaned if len(cleaned) >= 7 else None


# ===== AI Prompt =====

ANALYSIS_PROMPT_TEMPLATE = """You are an expert sales call analyst for a VILLA ACQUISITION company in India that signs up villa owners to list their properties for short-stay/vacation rentals.

The call is between:
- POC: salesperson from acquisition team
- OWNER: villa owner being pitched

Return ONLY valid JSON with these exact fields:

{
  "summary": "2-3 sentence summary in plain English",
  "call_stage": "first contact / discovery / pitch / negotiation / site visit booking / closing / follow-up",
  "villa_details": {
    "location": "City/area or 'Not mentioned'",
    "bedrooms": "Number or 'Not mentioned'",
    "amenities": ["pool, lawn, view, etc."],
    "property_status": "ready / under-construction / existing rental / Not mentioned",
    "any_other_notable_features": "USPs or unique aspects, if any"
  },
  "owner_profile": {
    "current_usage": "personal use / already renting / vacant / Not mentioned",
    "existing_tie_ups": "Competitor brand or 'None mentioned'",
    "decision_maker": "Yes / No / Unclear",
    "interest_level": "high / medium / low / cannot tell"
  },
  "commercials_discussed": {
    "revenue_expectation": "Owner's expectation including exact numbers/percentages, or 'Not discussed'",
    "revenue_share_or_model_mentioned": "What POC offered including exact percentages (e.g. '75% to owner, 25% to company'), or 'Not discussed'",
    "contract_or_exclusivity": "Mention of exclusivity/lock-in/contract length"
  },
  "owner_concerns": ["specific worries: damage, brand trust, loss of personal access, etc."],
  "owner_requirements": ["specific things owner wants: minimum guarantee, control over bookings, blocked dates"],
  "commitments_made": ["specific commitments by either side"],
  "next_action": "Concrete next step with timeline",
  "sentiment": "positive / neutral / negative",
  "sentiment_reason": "One line",
  "call_outcome": "interested / not interested / needs follow-up / site visit booked / ready to sign / closed-won / closed-lost",
  "quality_score": {
    "introduction_and_rapport": {"score": 0, "reason": "one line"},
    "discovery_of_property": {"score": 0, "reason": "one line"},
    "discovery_of_owner_needs": {"score": 0, "reason": "one line"},
    "pitch_clarity": {"score": 0, "reason": "one line"},
    "objection_handling": {"score": 0, "reason": "one line"},
    "commercials_handling": {"score": 0, "reason": "one line"},
    "clear_next_step": {"score": 0, "reason": "one line"},
    "professionalism_and_language": {"score": 0, "reason": "one line"},
    "overall_score": 0
  },
  "coaching_feedback": "2-3 sentences of constructive feedback for the POC"
}

Rules:
- Scores are integers 0-10. Replace 0 above with actual score.
- IMPORTANT: Capture ALL specific numbers, percentages, dates and amounts mentioned in the call. Do not skip them.
- owner_concerns vs owner_requirements: do NOT duplicate.
- Use empty arrays [] if nothing applies.
- All values in clear English even if call was in another language.

Transcript:
__TRANSCRIPT__

Return ONLY JSON. No other text."""


# ===== Streamlit UI =====

st.set_page_config(page_title="Villa Acquisition AI", page_icon="🏡", layout="wide")
st.title("🏡 Villa Acquisition Call Analyzer")
st.caption("Upload calls → AI analyzes → auto-pushes to Freshsales. Single file or bulk ZIP.")

# ===== MODE TOGGLE =====
mode = st.radio("Upload Mode", ["Single Call", "Bulk ZIP"], horizontal=True)
st.divider()

# ==============================================================
# SINGLE CALL MODE
# ==============================================================
if mode == "Single Call":
    with st.sidebar:
        st.header("Upload Call")
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=["mp3", "wav", "m4a", "ogg", "flac", "mpeg", "mpg"]
        )
        phone_number = st.text_input("Phone Number (with country code)", placeholder="+918452956426")
        poc_name = st.text_input("POC Name (optional)", placeholder="e.g. Rohan")
        analyze_btn = st.button("🚀 Analyze & Push to Freshsales", type="primary", use_container_width=True)

    if analyze_btn:
        if uploaded_file is None:
            st.warning("Please upload an audio file.")
            st.stop()
        if not phone_number.strip():
            st.warning("Please enter the phone number.")
            st.stop()

        temp_path = "temp_" + uploaded_file.name
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        compressed_path = "compressed_audio.mp3"
        try:
            compress_audio(temp_path, compressed_path)
        except Exception as e:
            st.error("Audio compression failed: " + str(e))
            os.remove(temp_path)
            st.stop()

        duration_seconds = get_duration(compressed_path)
        os.remove(temp_path)

        with st.spinner("Looking up contact in Freshsales..."):
            contact = find_contact_by_phone(phone_number.strip())
        if not contact:
            st.error("No Freshsales contact found for " + phone_number)
            os.remove(compressed_path)
            st.stop()

        contact_id = int(contact["id"])
        contact_name = contact.get("name") or contact.get("display_name", "N/A")
        st.success("Contact found: **" + contact_name + "** (ID: " + str(contact_id) + ")")

        with st.spinner("Routing: Deal → Property → Contact..."):
            target_type, target_id, target_name = find_routing_target(contact_id)

        label_map = {"deal": "Deal", "sales_account": "Property", "contact": "Contact"}
        nice_type = label_map.get(target_type, target_type)
        target_label = target_name or contact_name

        if target_type == "deal":
            st.info("Routing to **Deal**: " + str(target_label))
        elif target_type == "sales_account":
            st.info("Routing to **Property**: " + str(target_label))
        else:
            st.warning("Routing to **Contact**: " + contact_name)

        with st.spinner("Transcribing & translating audio..."):
            transcript = transcribe(compressed_path)

        with st.spinner("AI analyzing the call..."):
            analysis = analyze(transcript)

        os.remove(compressed_path)

        with st.spinner("Pushing to Freshsales..."):
            ok_call = push_phone_call(target_type, target_id, phone_number.strip())
            note_html = build_note_html(analysis, contact_name, duration_seconds)
            ok_note = push_note(target_type, target_id, note_html)

        if ok_call and ok_note:
            st.success("✅ Pushed to " + nice_type + ": **" + str(target_label) + "**")
        elif ok_note:
            st.warning("Note pushed but phone call failed.")
        elif ok_call:
            st.warning("Phone call pushed but note failed.")
        else:
            st.error("Failed to push to Freshsales.")

        st.divider()
        tab1, tab2, tab3 = st.tabs(["📋 CRM Note Preview", "🎯 QA Report", "📝 Full Transcript"])

        with tab1:
            st.markdown("**POC:** " + (poc_name or "N/A") + " | **Owner:** " + contact_name +
                        " | **Stage:** `" + str(analysis.get('call_stage', 'N/A')) +
                        "` | **Outcome:** `" + str(analysis.get('call_outcome', 'N/A')) + "`")
            st.divider()
            st.markdown("### Summary")
            st.write(analysis.get("summary", "—"))
            col1, col2 = st.columns(2)
            with col1:
                v = analysis.get("villa_details", {})
                st.markdown("### 🏠 Villa Details")
                st.markdown("- **Location:** " + str(v.get('location', 'N/A')))
                st.markdown("- **Bedrooms:** " + str(v.get('bedrooms', 'N/A')))
                st.markdown("- **Status:** " + str(v.get('property_status', 'N/A')))
                am = v.get("amenities", [])
                st.markdown("- **Amenities:** " + (', '.join(am) if am else 'Not mentioned'))
                o = analysis.get("owner_profile", {})
                st.markdown("### 👤 Owner Profile")
                st.markdown("- **Current usage:** " + str(o.get('current_usage', 'N/A')))
                st.markdown("- **Existing tie-ups:** " + str(o.get('existing_tie_ups', 'N/A')))
                st.markdown("- **Decision maker:** " + str(o.get('decision_maker', 'N/A')))
                st.markdown("- **Interest level:** " + str(o.get('interest_level', 'N/A')))
            with col2:
                st.markdown("### ⚠️ Owner Concerns")
                for c in analysis.get("owner_concerns", []) or ["None raised"]:
                    st.markdown("- " + str(c))
                st.markdown("### ✅ Owner Requirements")
                for r in analysis.get("owner_requirements", []) or ["None mentioned"]:
                    st.markdown("- " + str(r))
                comm = analysis.get("commercials_discussed", {})
                st.markdown("### 💰 Commercials")
                st.markdown("- **Owner expectation:** " + str(comm.get('revenue_expectation', 'N/A')))
                st.markdown("- **Model offered:** " + str(comm.get('revenue_share_or_model_mentioned', 'N/A')))
                st.markdown("- **Contract terms:** " + str(comm.get('contract_or_exclusivity', 'N/A')))
            st.divider()
            st.markdown("### 🎯 Next Action")
            st.info(analysis.get("next_action", "—"))

        with tab2:
            qs = analysis.get("quality_score", {})
            overall = qs.get("overall_score", 0)
            if overall >= 8:
                st.success("### Overall Score: " + str(overall) + "/10  ⭐")
            elif overall >= 5:
                st.warning("### Overall Score: " + str(overall) + "/10")
            else:
                st.error("### Overall Score: " + str(overall) + "/10")
            sent = analysis.get("sentiment", "neutral")
            sent_emoji = {"positive": "😊", "neutral": "😐", "negative": "😟"}.get(sent.lower(), "")
            st.markdown("**Sentiment:** " + sent_emoji + " " + sent.title() + " — _" + analysis.get("sentiment_reason", "") + "_")
            st.divider()
            for p in ["introduction_and_rapport", "discovery_of_property", "discovery_of_owner_needs",
                      "pitch_clarity", "objection_handling", "commercials_handling",
                      "clear_next_step", "professionalism_and_language"]:
                entry = qs.get(p, {})
                score = entry.get("score", 0)
                st.markdown("**" + p.replace("_", " ").title() + "** — `" + str(score) + "/10`")
                st.progress(score / 10)
                st.caption(entry.get("reason", ""))
            st.divider()
            st.markdown("### 💡 Coaching Feedback")
            st.info(analysis.get("coaching_feedback", "—"))

        with tab3:
            st.text_area("Transcript", transcript, height=400)

    else:
        st.info("👈 Upload a call recording, enter the phone number, and click Analyze.")


# ==============================================================
# BULK ZIP MODE
# ==============================================================
else:
    st.markdown("""
    ### How bulk upload works
    - ZIP file containing audio files (mp3, wav, m4a, etc.)
    - **Each filename must be the phone number** e.g. `+917006347128.mp3`
    - Each file is processed one by one and pushed to Freshsales automatically
    - A summary table is shown at the end
    """)

    poc_name_bulk = st.text_input("POC Name (optional, applies to all calls)", placeholder="e.g. Rohan")
    zip_file = st.file_uploader("Upload ZIP file", type=["zip"])
    bulk_btn = st.button("🚀 Process All & Push to Freshsales", type="primary")

    if bulk_btn:
        if zip_file is None:
            st.warning("Please upload a ZIP file.")
            st.stop()

        # Extract audio files from ZIP
        with zipfile.ZipFile(io.BytesIO(zip_file.read())) as zf:
            audio_files = [
                name for name in zf.namelist()
                if not name.startswith("__MACOSX")
                and name.lower().endswith(AUDIO_EXTENSIONS)
            ]
            if not audio_files:
                st.error("No audio files found in the ZIP. Make sure files end in .mp3 / .wav / .m4a etc.")
                st.stop()

            st.info(f"Found **{len(audio_files)} audio files** in ZIP. Processing...")
            st.divider()

            # Results tracking
            results = []

            for idx, filename in enumerate(audio_files):
                short_name = os.path.basename(filename)
                phone = phone_from_filename(short_name)

                st.markdown(f"### [{idx+1}/{len(audio_files)}] `{short_name}`")

                if not phone:
                    st.warning("⚠️ Could not extract phone number from filename — skipping.")
                    results.append({"file": short_name, "phone": "?", "contact": "—", "status": "❌ Bad filename"})
                    continue

                st.caption("📞 Phone: " + phone)

                # Write audio to temp file
                temp_path = "bulk_temp_" + short_name.replace("+", "")
                with open(temp_path, "wb") as f:
                    f.write(zf.read(filename))

                # Compress
                compressed_path = "bulk_compressed.mp3"
                try:
                    compress_audio(temp_path, compressed_path)
                    os.remove(temp_path)
                except Exception as e:
                    st.error("Compression failed: " + str(e))
                    os.remove(temp_path)
                    results.append({"file": short_name, "phone": phone, "contact": "—", "status": "❌ Compress failed"})
                    continue

                duration_seconds = get_duration(compressed_path)

                # Find contact
                contact = find_contact_by_phone(phone)
                if not contact:
                    st.warning("⚠️ No contact found in Freshsales for " + phone + " — skipping.")
                    os.remove(compressed_path)
                    results.append({"file": short_name, "phone": phone, "contact": "Not found", "status": "⚠️ Skipped"})
                    continue

                contact_id = int(contact["id"])
                contact_name = contact.get("name") or contact.get("display_name", "N/A")
                st.success("✅ Contact: **" + contact_name + "**")

                # Route
                target_type, target_id, target_name = find_routing_target(contact_id)
                target_label = target_name or contact_name
                label_map = {"deal": "Deal", "sales_account": "Property", "contact": "Contact"}
                st.caption("📍 Routing to " + label_map.get(target_type, target_type) + ": " + str(target_label))

                # Transcribe
                with st.spinner("Transcribing..."):
                    try:
                        transcript = transcribe(compressed_path)
                    except Exception as e:
                        st.error("Transcription failed: " + str(e))
                        os.remove(compressed_path)
                        results.append({"file": short_name, "phone": phone, "contact": contact_name, "status": "❌ Transcription failed"})
                        continue

                # Analyze
                with st.spinner("Analyzing..."):
                    try:
                        analysis = analyze(transcript)
                    except Exception as e:
                        st.error("Analysis failed: " + str(e))
                        os.remove(compressed_path)
                        results.append({"file": short_name, "phone": phone, "contact": contact_name, "status": "❌ Analysis failed"})
                        continue

                os.remove(compressed_path)

                # Push to Freshsales
                ok_call = push_phone_call(target_type, target_id, phone)
                note_html = build_note_html(analysis, contact_name, duration_seconds)
                ok_note = push_note(target_type, target_id, note_html)

                if ok_call and ok_note:
                    status = "✅ Pushed"
                elif ok_note:
                    status = "⚠️ Note only"
                elif ok_call:
                    status = "⚠️ Call only"
                else:
                    status = "❌ Push failed"

                st.markdown("**Result:** " + status + " | Outcome: `" + str(analysis.get("call_outcome", "—")) + "` | Score: `" + str(analysis.get("quality_score", {}).get("overall_score", "—")) + "/10`")
                results.append({
                    "file": short_name,
                    "phone": phone,
                    "contact": contact_name,
                    "outcome": analysis.get("call_outcome", "—"),
                    "score": analysis.get("quality_score", {}).get("overall_score", "—"),
                    "next_action": analysis.get("next_action", "—"),
                    "status": status
                })
                st.divider()

        # Summary table
        st.markdown("## 📊 Bulk Processing Summary")
        st.dataframe(results, use_container_width=True)
        done = sum(1 for r in results if r["status"].startswith("✅"))
        skipped = sum(1 for r in results if r["status"].startswith("⚠️"))
        failed = sum(1 for r in results if r["status"].startswith("❌"))
        col1, col2, col3 = st.columns(3)
        col1.metric("✅ Pushed", done)
        col2.metric("⚠️ Skipped", skipped)
        col3.metric("❌ Failed", failed)
