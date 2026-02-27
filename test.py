from bot import Seeker, Oracle
from openai import OpenAI
import os
from country import country_choice

attribute_blank = None

ATTRIBUTE_SPACE = [
    "continent",
    "sub_region",
    "hemisphere",
    "landlocked",
    "is_island",
    "population_band",
    "climate_band",
    "gdp_per_capita",
    "altitude",
    "terrain"
]
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

seeker = Seeker(
    client = client,
    model="gpt-5-nano",
    question_budget=5,
    attribute_space=ATTRIBUTE_SPACE
)

oracle = Oracle(
    client = client,
    model="gpt-5-nano",
    country_choice=country_choice,
    question_budget=5,
    attribute_space=attribute_blank
)

print(f"The chosen country: {oracle.hidden_country}")
print("Game starting...\n")

while seeker.questions_asked < seeker.question_budget:
    question = seeker.act()
    print(f"Seeker: {question}")

    oracle.receive_question(question)
    answer = oracle.act()
    print(f"Oracle: {answer}")

    seeker.update_history(question, answer)
    oracle.update_history(question, answer)

guess = seeker.make_guess()
print(f"\nSeeker's final guess: {guess}")

if guess == oracle.hidden_country:
    print("The seeker guessed correctly")
else:
    print("Incorrect. Better look next time")
