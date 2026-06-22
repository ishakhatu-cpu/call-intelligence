import os
import json
import requests
import html as html_lib
from requests.auth import HTTPBasicAuth
from datetime import datetime, timezone
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from mutagen import File as MutagenFile

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

FS_DOMAIN = os.getenv("FRESHSALES_DOMAIN")
FS_EMAIL = os.getenv("FS_ADMIN_EMAIL")
FS_PASSWORD = os.getenv("FS_ADMIN_PASSWORD")

FS_AUTH = HTTPBasicAuth(FS_EMAIL, FS_PASSWORD)
FS_HEADERS = {"Content-Type": "application/json"}


# ===== Freshsales helpers (Basic Auth) =====

def find_contact_by_phone(phone):
    url = f"https://{FS_DOMAIN}/crm/sales/api/lookup"
    params = {"q": phone.replace("+", "%2B"), "f": "mobile_number", "entities": "contact"}
    r = requests.get(url, headers=FS_HEADERS, params=params, auth=FS_AUTH)
    if r.status_code != 200:
        return None
    for c in r.json().get("contacts", {}).get("contacts", []):
        if c.get("mobile_number") == phone:
            return c
    return None


def find_routing_target(contact_id):
    """
    Decide where the call log should go:
      1. Deal (if exists)
      2. Property / SalesAccount (if exists, no deal)
      3. Contact (last resort)
    Returns: (target_type, target_id, target_name)
    """
    url = f"https://{FS_DOMAIN}/crm/sales/api/contacts/{contact_id}?include=deals,sales_account"
    r = requests.get(url, headers=FS_HEADERS, auth=FS_AUTH)
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
    """Push a phone_call activity using Runo-style payload + Basic Auth."""
    payload = {
        "phone_call": {
            "call_direction": True,
            "targetable_type": target_type,
            "targetable": {"id": target_id, "phone": phone}
        }
    }
    r = requests.post(f"https://{FS_DOMAIN}/crm/sales/api/phone_calls",
                      headers=FS_HEADERS, json=payload, auth=FS_AUTH)
    return r.status_code in (200, 201)


def push_note(target_type, target_id, html_body):
    """Push a separate Note for the AI analysis. Note targetable_type expects
    PascalCase like 'Deal' / 'SalesAccount' / 'Contact' (not lowercase)."""
    type_map = {"deal": "Deal", "sales_account": "SalesAccount", "contact": "Contact"}
    payload = {
        "note": {
            "description": html_body,
            "targetable_type": type_map.get(target_type, target_type),
            "targetable_id": target_id,
        }
    }
    r = requests.post(f"https://{FS_DOMAIN}/crm/sales/api/notes",
                      headers=FS_HEADERS, json=payload, auth=FS_AUTH)
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
st.caption("Upload an owner call -> AI analyzes -> auto-pushes phone call + note to Freshsales. Routes to Deal -> Property -> Contact.")

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
        st.warning("Please enter the phone number for this call.")
        st.stop()

    temp_path = "temp_" + uploaded_file.name
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        from pydub import AudioSegment
        audio_segment = AudioSegment.from_file(temp_path)
        fixed_path = temp_path.replace(".mp3", "_fixed.mp3")
        audio_segment.export(fixed_path, format="mp3")
        temp_path = fixed_path
        audio = MutagenFile(temp_path)
    except Exception:
        audio = None
    duration_seconds = int(audio.info.length) if audio and audio.info else 60

    with st.spinner("Looking up contact in Freshsales..."):
        contact = find_contact_by_phone(phone_number.strip())
    if not contact:
        st.error("No Freshsales contact found with mobile_number = " + phone_number)
        os.remove(temp_path)
        st.stop()

    contact_id = contact["id"]
    contact_name = contact.get("display_name", "N/A")
    st.success("Contact found: **" + contact_name + "** (ID: " + str(contact_id) + ")")

    with st.spinner("Routing: checking Deal -> Property -> Contact..."):
        target_type, target_id, target_name = find_routing_target(contact_id)

    label_map = {"deal": "Deal", "sales_account": "Property", "contact": "Contact"}
    nice_type = label_map.get(target_type, target_type)
    target_label = target_name or contact_name

    if target_type == "deal":
        st.info("Routing to **Deal**: " + str(target_label))
    elif target_type == "sales_account":
        st.info("No deal found. Routing to **Property**: " + str(target_label))
    else:
        st.warning("No deal or property linked. Routing to **Contact**: " + contact_name)

    with st.spinner("Transcribing & translating audio to English..."):
        with open(temp_path, "rb") as file:
            transcription = client.audio.translations.create(
                file=(uploaded_file.name, file.read()),
                model="whisper-large-v3",
            )
        transcript = transcription.text

    with st.spinner("AI analyzing the call..."):
        prompt = ANALYSIS_PROMPT_TEMPLATE.replace("__TRANSCRIPT__", transcript)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        analysis = json.loads(response.choices[0].message.content)

    os.remove(temp_path)

    push_status = st.empty()
    with st.spinner("Pushing phone_call + note to Freshsales..."):
        ok_call = push_phone_call(target_type, target_id, phone_number.strip())
        note_html = build_note_html(analysis, contact_name, duration_seconds)
        ok_note = push_note(target_type, target_id, note_html)

    if ok_call and ok_note:
        push_status.success("✅ Phone call + note pushed to " + nice_type + ": **" + str(target_label) + "**")
    elif ok_note:
        push_status.warning("Note pushed but phone call failed.")
    elif ok_call:
        push_status.warning("Phone call pushed but note failed.")
    else:
        push_status.error("Failed to push to Freshsales. Check credentials.")

    st.divider()

    tab1, tab2, tab3 = st.tabs(["📋 CRM Note Preview", "🎯 QA Report (Manager View)", "📝 Full Transcript"])

    with tab1:
        st.subheader("Note pushed to Freshsales")
        st.markdown("**POC:** " + (poc_name or "N/A") + " | **Owner:** " + contact_name + " | **Stage:** `" + str(analysis.get('call_stage', 'N/A')) + "` | **Outcome:** `" + str(analysis.get('call_outcome', 'N/A')) + "`")
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
            concerns = analysis.get("owner_concerns", [])
            if concerns:
                for c in concerns:
                    st.markdown("- " + str(c))
            else:
                st.caption("None raised")

            st.markdown("### ✅ Owner Requirements")
            reqs = analysis.get("owner_requirements", [])
            if reqs:
                for r in reqs:
                    st.markdown("- " + str(r))
            else:
                st.caption("None mentioned")

            comm = analysis.get("commercials_discussed", {})
            st.markdown("### 💰 Commercials Discussed")
            st.markdown("- **Owner expectation:** " + str(comm.get('revenue_expectation', 'N/A')))
            st.markdown("- **Model offered:** " + str(comm.get('revenue_share_or_model_mentioned', 'N/A')))
            st.markdown("- **Contract terms:** " + str(comm.get('contract_or_exclusivity', 'N/A')))

        st.divider()
        st.markdown("### 🎯 Next Action")
        st.info(analysis.get("next_action", "—"))

    with tab2:
        st.subheader("Quality Assurance Report (Internal)")
        qs = analysis.get("quality_score", {})
        overall = qs.get("overall_score", 0)
        if overall >= 8:
            st.success("### Overall Score: " + str(overall) + "/10  ⭐")
        elif overall >= 5:
            st.warning("### Overall Score: " + str(overall) + "/10")
        else:
            st.error("### Overall Score: " + str(overall) + "/10")

        sent = analysis.get("sentiment", "neutral")
        sent_reason = analysis.get("sentiment_reason", "")
        sent_emoji = {"positive": "😊", "neutral": "😐", "negative": "😟"}.get(sent.lower(), "")
        st.markdown("**Owner Sentiment:** " + sent_emoji + " " + sent.title() + " — _" + sent_reason + "_")

        st.divider()
        st.markdown("### Score Breakdown")
        params = ["introduction_and_rapport", "discovery_of_property", "discovery_of_owner_needs",
                  "pitch_clarity", "objection_handling", "commercials_handling",
                  "clear_next_step", "professionalism_and_language"]
        for p in params:
            entry = qs.get(p, {})
            score = entry.get("score", 0)
            reason = entry.get("reason", "")
            label = p.replace("_", " ").title()
            st.markdown("**" + label + "** — `" + str(score) + "/10`")
            st.progress(score / 10)
            st.caption(reason)

        st.divider()
        st.markdown("### 💡 Coaching Feedback")
        st.info(analysis.get("coaching_feedback", "—"))

    with tab3:
        st.subheader("Full Transcript (auto-translated to English)")
        st.text_area("Transcript", transcript, height=400)

else:
    st.info("👈 Upload a call recording, enter the phone number, and click the button.")
