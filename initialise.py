from openai import OpenAI
from bot import Seeker, Oracle
from game_environment import GameEnvironment
from country import country_choice
import os
from attributes import ATTRIBUTE_SPACE

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

seeker = Seeker(
    client=client,
    model="gpt-5-nano",
    question_budget=5,
    attribute_space=ATTRIBUTE_SPACE,
)

oracle = Oracle(
    client = client,
    model="gpt-5-nano",
    country_choice=country_choice,
    question_budget=5,
    attribute_space=ATTRIBUTE_SPACE,
)

game = GameEnvironment(seeker, oracle)
seeker.game = game
oracle.game = game

game.run()

#print(game.result())
