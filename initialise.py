from openai import OpenAI
from bot import Seeker, Oracle
from game_environment import GameEnvironment
from country import country_choice
import os
from attributes import ATTRIBUTE_SPACE

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#Parameters
model = "gpt-5-nano"
questions = 8

seeker = Seeker(
    client=client,
    model=model,
    question_budget=questions,
    attribute_space=ATTRIBUTE_SPACE,
)

oracle = Oracle(
    client = client,
    model=model,
    country_choice=country_choice,
    question_budget=questions,
    attribute_space=ATTRIBUTE_SPACE,
)

game = GameEnvironment(seeker, oracle)
seeker.game = game
oracle.game = game

game.run()

#print(game.result())
