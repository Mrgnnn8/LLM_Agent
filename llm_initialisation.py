import os
import json
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM = """You are a country-guessing Seeker agent.

You must respond ONLY in valid JSON, with ONE of these formats:
1) Ask:
{"action":"ASK","attribute":"<one of attribute_space>"}

2) Guess:
{"action":"GUESS","country":"<country name>"}

Rules:
- You have a limited number of questions.
- If questions_left == 0, you MUST output a GUESS.
- Do not invent attributes outside attribute_space.
- Keep your ASK concise.
"""

ATTRIBUTE_SPACE = [
    "continent",
    "sub_region",
    "hemisphere",
    "landlocked",
    "is_island",
    "population_band",
    "climate_band",
    "language_family"
]

MEMORY = []

STATE = {
    "turn": 0,
    "questions_left": 3,
    "known_constraints": {}
}

def build_messages(user_text: str):
    """Builds the message list sent to the model this turn."""
    observation = {
        "turn": STATE["turn"],
        "questions_left": STATE["questions_left"],
        "attribute_space": ATTRIBUTE_SPACE,
        "known_constraints": STATE["known_constraints"],
        "legal_actions": ["GUESS"] if STATE["questions_left"] == 0 else ["ASK", "GUESS"]
    }

    messages = MEMORY.copy()

    messages.append({
        "role": "user",
        "content": (
            f"USER_INPUT:\n{user_text}\n\n"
            f"CURRENT_GAME_STATE_JSON:\n{json.dumps(observation, indent=2)}"
        )
    })

    return messages

def ask(user_text: str) -> str:
    STATE["turn"] += 1

    resp = client.responses.create(
        model="gpt-5",
        instructions=SYSTEM,
        input=build_messages(user_text)
    )

    assistant_text = (resp.output_text or "").strip()

    MEMORY.append({"role": "user", "content": user_text})
    MEMORY.append({"role": "assistant", "content": assistant_text})

    try:
        action = json.loads(assistant_text)
        if action.get("action") == "ASK" and STATE["questions_left"] > 0:
            STATE["questions_left"] -= 1
    except json.JSONDecodeError:
        pass

    return assistant_text

if __name__ == "__main__":
    while True:
        msg = input("You: ").strip()
        if not msg:
            continue
        if msg.lower() in {"exit", "quit"}:
            break
        print("AI:", ask(msg))
