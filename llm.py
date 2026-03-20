import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================================================
# CONFIG
# =========================================================
FEATURE_JSON_PATH = "/content/final_integrated_feature_summary.json"
OUTPUT_REPORT_PATH = "/content/final_llm_interview_report.json"

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

MAX_NEW_TOKENS = 1200

# =========================================================
# PROMPT
# =========================================================
def build_system_prompt() -> str:
    return """Persona: You are a Behavioral Communications Expert and Interview Coach. Your goal is to translate raw biomechanical data into a human narrative that describes a candidate’s interpersonal presence, professional warmth, and self-regulation.

Your role is NOT just to analyze data, but to give HUMAN-CENTRIC feedback like a real interviewer or mentor.

You will receive structured signals derived from:
- Facial behavior (OpenFace)
- Body posture & gestures (BlazePose)
- Attention / gaze / stability metrics

Your job is to convert these signals into:
- Clear human observations
- Practical interview feedback
- Coaching-style insights

IMPORTANT BEHAVIOR:
- Speak like a mentor, not a machine
- Avoid technical terms like "pose_Rx", "AU12_r", etc.
- Translate numbers into human meaning
- Use phrases like:
  - "You appear..."
  - "It seems..."
  - "This suggests..."
  - "Could be improved by..."

Human-Centric Guidelines:
Interpret the Journey: Use the "timeline" data to see if the candidate evolved. For example, did they start with "settling-in nerves" and transition into "focused engagement"?
Focus on "Soft" Indicators: - Translate Gaze/Attention into "Interpersonal Presence" and "Active Listening."
Translate Action Units (AU) into "Professional Warmth" vs. "Concentrated Tension."
Translate Pose/Stability into "Composure" and "Openness."
Empathetic Language: Use observational phrases like "The candidate appears to...", "There is a tendency toward...", or "The candidate displays a listening posture that..."
Avoid Clinical Jargon: Do not just list AU numbers. Explain that "lip tightening" may indicate a moment of deep thought or slight stress.
Contextualize Nerves: Treat initial instability as a natural human response to an interview start rather than a permanent character flaw.

DO NOT:
- Sound robotic
- Dump raw metrics
- Overclaim psychological certainty

You are evaluating:
- Confidence
- Eye contact
- Facial expressiveness
- Head movement
- Posture
- Gesture usage

FINAL GOAL:
Give a realistic interview-style evaluation of the candidate.

Return ONLY valid JSON.
""".strip()

def build_user_prompt(feature_data: dict) -> str:
    return f"""
Analyze the following structured interview video feature summary and generate a human-centric evaluation report.
Convert all signals into clear, meaningful observations and coaching-style feedback.

Return JSON in the following format:

{{
  "overall_summary": "Brief summary of overall performance",

  "confidence": {{
    "score": 0,
    "label": "Very Low | Low | Moderate | High | Very High",
    "insight": "Short explanation of confidence level"
  }},

  "body_language_analysis": [
    {{
      "signal": "Eye Contact",
      "observation": "Observed behavior",
      "impact": "Interpretation and suggested improvement"
    }},
    {{
      "signal": "Facial Expression",
      "observation": "Observed behavior",
      "impact": "Interpretation and suggested improvement"
    }},
    {{
      "signal": "Posture",
      "observation": "Observed behavior",
      "impact": "Interpretation and suggested improvement"
    }},
    {{
      "signal": "Gestures",
      "observation": "Observed behavior",
      "impact": "Interpretation and suggested improvement"
    }}
  ],

  "key_insight": "Single important takeaway about the candidate",

  "strengths": [
    "Strength point",
    "Strength point"
  ],

  "improvements": [
    "Improvement point",
    "Improvement point"
  ],

  "action_plan": [
    "Actionable recommendation",
    "Actionable recommendation"
  ],

  "final_verdict": {{
    "confidence_score": 0,
    "interview_readiness": "percentage",
    "summary": "Final evaluation of candidate readiness"
  }},

  "caution_note": "Short disclaimer about interpretation limitations"
}}

Ensure:
- Observations are based only on the provided data
- Language is natural and easy to understand
- Feedback is practical and interview-focused
- No technical terms or raw feature names are exposed

Feature summary:
{json.dumps(feature_data, indent=2)}
""".strip()

# =========================================================
# MODEL LOAD
# =========================================================
def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    return tokenizer, model

# =========================================================
# GENERATION
# =========================================================
def generate_report(tokenizer, model, system_prompt: str, user_prompt: str) -> dict:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # Try parsing direct JSON
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        # Fallback: try extracting JSON block
        start = response_text.find("{")
        end = response_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = response_text[start:end + 1]
            return json.loads(candidate)
        raise ValueError(f"Model output was not valid JSON:\n{response_text}")

# =========================================================
# MAIN
# =========================================================
def main():
    if not os.path.exists(FEATURE_JSON_PATH):
        raise FileNotFoundError(f"Feature JSON not found: {FEATURE_JSON_PATH}")

    with open(FEATURE_JSON_PATH, "r", encoding="utf-8") as f:
        feature_data = json.load(f)

    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(feature_data)

    tokenizer, model = load_model(MODEL_NAME)
    report = generate_report(tokenizer, model, system_prompt, user_prompt)

    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nSaved LLM report to: {OUTPUT_REPORT_PATH}")

if __name__ == "__main__":
    main()
