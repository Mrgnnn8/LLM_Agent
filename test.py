from bot import Seeker
from openai import OpenAI
import os


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


seeker = Seeker(
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")),
    model="gpt-5",
    question_budget=10,
    attribute_space=ATTRIBUTE_SPACE
)

Print("Think of a country. Press enter when ready.")
input()

while seeker.questions_asked < seeker.question_budget:
    question = seeker.act()
    print(f"\nSeeker: {question}")

    answer = input("Your answer: ")
    seeker.update_history(question, answer)

guess = seeker.make_guess()
print(f"\nSeeker's final guess: {guess}")

