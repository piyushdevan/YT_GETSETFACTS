"""
Script Generator Module.

Uses Google Gemini API to:
1. Auto-generate video topics when the manual queue is empty.
2. Generate structured scripts, catchy titles, descriptions, and tags.
"""
import json
from typing import Dict, Any
from google import genai
import config

_client = genai.Client(api_key=config.GEMINI_API_KEY)
_MODEL = "gemini-3.5-flash-lite"


def _extract_text(response) -> str:
    """Extracts text content from model response without non-text part warnings."""
    if hasattr(response, "candidates") and response.candidates:
        candidate = response.candidates[0]
        if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
            text_parts = [part.text for part in candidate.content.parts if hasattr(part, "text") and part.text]
            if text_parts:
                return "".join(text_parts).strip()
    return response.text.strip() if hasattr(response, "text") and response.text else ""


def generate_topic(video_format: str) -> str:
    """Generates a unique catchy video topic idea for the specified format, avoiding duplicates."""
    import db_manager
    length_hint = "a short, punchy fact" if video_format == "short" else "a broader, multi-part topic"
    past_topics = db_manager.get_past_topics_for_format(video_format)
    exclusion_text = ""
    if past_topics:
        past_list_str = "\n".join(f"- {t}" for t in past_topics[:100])
        exclusion_text = f"\nDO NOT suggest any of these previously covered topics for {video_format}:\n{past_list_str}\n"

    for attempt in range(10):
        prompt = f"""
You generate ONE FRESH curiosity-driven topic with strong potential for broad YouTube audience interest.
Niche: {config.CHANNEL_NICHE}
TOPIC SELECTION RULES:
- Prefer topics connected to recent events, discoveries, records, sightings,
  missions, research, or developments.
- Prefer subjects that ordinary people can understand immediately.
- Prefer topics that can be explained with compelling visuals.
- Prefer topics that create an immediate "Wait, what?" reaction.
- Prefer surprising, rare, extreme, mysterious, unusual, or record-breaking subjects.
- The topic must have a strong curiosity gap.
- Avoid topics that require specialist knowledge.
- Avoid boring textbook-style educational topics.
- Avoid generic "facts about X" topics.
- Avoid topics based primarily on explanations of how ordinary systems work.

Provide ONE single unique topic idea suitable for {length_hint}.{exclusion_text}
Reply with ONLY the topic title, nothing else. No quotes, no numbering.
"""
        response = _client.models.generate_content(model=_MODEL, contents=prompt)
        topic = _extract_text(response).strip('"\n ')
        if topic and not db_manager.is_topic_duplicate(topic, video_format):
            return topic

    return topic


def generate_script(topic: str, video_format: str) -> Dict[str, Any]:
    """
    Generates structured script metadata for a topic.
    Returns: {"title": str, "script": str, "description": str, "tags": list}
    """
    if video_format == "short":
        word_count = "100-150 words, fast-paced, strong hook in the first sentence"
    else:
        word_count = "1000-1500 words, structured with an introduction, 6-7 key points, and a conclusion"

    prompt = f"""
Write a YouTube video script about: "{topic}"

Requirements:
- Length: {word_count}
- Tone: Engaging, conversational spoken tone for voiceover delivery
- Exclude stage directions, music tags, or section headers (only spoken content)
- Start with a compelling hook sentence.
- The first 1-2 sentences must create immediate curiosity, surprise, or disbelief.
- Do NOT begin with generic phrases like "Did you know", "In today's video", "Here's something interesting", or "Have you ever wondered".
- Reveal information progressively; do not give away the main answer immediately.
- Every 1-2 sentences should introduce a new fact, twist, consequence, or curiosity gap.
- Prioritize the most interesting and relevant facts; remove anything that slows the story down.
- Avoid unnecessary background information and filler.
- Build toward the most surprising fact or revelation near the end.
- The final 1-2 sentences should deliver a satisfying payoff.
- Use short, punchy sentences suitable for natural voiceover.
- Write for spoken delivery, not for reading.
- Avoid repetitive phrases and obvious AI-style wording.
- Do not exaggerate facts or create fake suspense.

    
Additional fields:
- Catchy YouTube title (under 80 characters)
- Short YouTube description (2-3 sentences with a subscribe call-to-action)
- 5 relevant video tags

Respond ONLY in valid JSON format:
{{
  "title": "...",
  "script": "...",
  "description": "...",
  "tags": ["...", "...", "...", "...", "..."]
}}
"""
    response = _client.models.generate_content(model=_MODEL, contents=prompt)
    raw = _extract_text(response)
    
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]

    data = json.loads(raw.strip())
    return data
