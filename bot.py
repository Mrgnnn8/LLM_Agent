import os
import json
import random
from openai import OpenAI
from abc import ABC, abstractmethod
from country import country_choice
from attributes import ATTRIBUTE_SPACE


class Brain(ABC):
    def __init__(self, client: str, role: str, question_budget: int, model: str, attribute_space: list):
        self.api_client = client
        self.model = model

        self.role = role
        self.history = [] #There is room to optimise how this list looks. It is not efficient right now.
        self.max_history = 10
        self.question_budget = question_budget
        self.questions_asked = 0
        self.questions_remaining = self.question_budget - self.questions_asked
        self.attribute_space = attribute_space
        self.country_choice = country_choice

    @abstractmethod
    def profile(self) -> str:
        # This function defines the role of the agent
        """
        Defines who the agent is and what its goal is.
        Returns a system prompt string.
        """
        pass

    def memory(self) -> str:
        # This function creates and stores the memory of the agent
        """
        Retrieves and formats the conversation history for the injection into the prompt.
        Respects max_history to avoid overflowing context window.
        """
        recent = self.history[-self.max_history:]
        if not recent:
            return "No questions have been asked yet."

        formatted = []
        for i, exchange in enumerate(recent, 1):
            formatted.append(f"Q{i}: {exchange['question']}")
            formatted.append(f"A{i}: {exchange['answer']}")
        return "\n".join(formatted)

    @abstractmethod
    def planning(self, context: str, history: str) -> str:
        # This function creates a plan for the agent
        """
        Reasons about what to do next given profile context and memory.
        Makes a separate LLM call so reasoning is observable and loggable.
        Returns a plan string that is passed to the action module.
        """
        pass

    @abstractmethod
    def action(self, plan: str) -> str:
        # This function creates an action for the agent
        """
        Takes the plan and produces the actual game output.
        Returns a question (Seeker) or answer (Oracle).
        """
        pass

    def act(self) -> str:
        """
        Orchestrates the four modules in sequence to build a prompt which is fed into the model.
        """
        context = self.profile()
        history = self.memory()
        self.last_plan = self.planning(context, history)
        output = self.action(self.last_plan)
        return output
    
    def call_llm(self, input: str) -> str:
        response = self.api_client.responses.create(
            model=self.model,
            instructions=self.profile(),
            input=input
        )
        response = response.output_text.strip()
        return response

    def update_history(self, question: str, answer: str): 
        self.history.append({"question": question, "answer": answer})


class Seeker(Brain):
    def __init__(self, client: OpenAI, model: str, question_budget: int, attribute_space: list):
        super().__init__(
            client=client,
            role="seeker",
            model=model,
            question_budget=question_budget,
            attribute_space=attribute_space,
        )
        self.candidate_count = len(self.country_choice)

    def profile(self) -> str:
        budget_remaining = self.question_budget - self.questions_asked
        attributes = (
            f"You may only ask questions about the following attributes: {', '.join(self.attribute_space)} "
            if self.attribute_space else
            "You may ask yes/no questions about the country. "
        )
        return (
            f"You are a strategic question-asker trying to identify a hidden country. "
            f"Your goal is to identify the country in as few questions as possible. "
            f"{attributes} "
            f"You have {budget_remaining} questions remaining out of {self.question_budget}. "
            f"The current score is {self.game.score}. "
            f"You want to reduce the amount of candidates remaining as much as possible to help minimise the score. "
            f"Be careful as you don't want to guess wrong and score 0. "
            f"Removing too many countries could lead to you removing countries that are the correct answer. "
        )

    def planning(self, context: str, history: str) -> str:
        current_candidates = self.candidate_list()
        
        user = (
            f"Game history so far:\n{history}\n\n "
            f"The current score is {self.game.score}. You want to reduce the amount of candidates remaining as much as possible to help minimise the score. "
            f"You have {self.question_budget - self.questions_asked} questions remaining.\n\n"
            f"Based on the current chat history, reason through the following steps:\n"
            f"1. What do you know so far about the country?\n"
            f"2. Given what you know, list the countries that are still possible from this list: {current_candidates}\n"
            f"3. How many candidates remain?\n"
            f"4. What is your strategy for your next question to eliminate the most candidates?\n\n"
            f"Format your response exactly like this:\n"
            f"REASONING: <your reasoning>\n"
            f"CANDIDATES: <comma seperated list of remaining countries>\n"
            f"COUNT: <number of candidates>\n"
            f"STRATEGY: <your next question strategy>\n"
        )
        plan = self.call_llm(user)
        return plan

    def action(self, plan: str) -> str:
        
        try:
            count_line = [line for line in plan.split("\n") if line.startswith("COUNT:")][0]
            self.candidate_count = int(count_line.replace("COUNT:", "").strip())
        except (IndexError, ValueError):
            self.candidate_count = len(self.country_choice)

        if self.candidate_count <= 2:
            return None

        user = (
            f"Your reasoning and strategy:\n{plan}\n\n"
            f"Based on your strategy, output your next yes/no question. "
            f"Just the question, no explanation"
        )

        self.questions_asked += 1
        return self.call_llm(user)

    def make_guess(self) -> str:
        user = (
            f"Game history:\n{self.memory()}\n\n "
            f"Based on everything you know, what is your final guess for the country? "
            f"Respond with only the country name. "
        )
        return self.call_llm(user)

    def candidate_list(self) -> list:
        if not os.path.exists("candidate_log.txt") or os.path.getsize("candidate_log.txt") == 0:
            return self.country_choice

        with open("candidate_log.txt", "r") as f:
            last_line = f.readlines()[-1].strip()
        
        return last_line.split(", ") if last_line else self.country_choice


class Oracle(Brain):
    def __init__(self, client: OpenAI, model: str, question_budget: int, country_choice: list, attribute_space: list):
        super().__init__(
            client=client,
            role="oracle",
            model=model,
            question_budget=question_budget,
            attribute_space=attribute_space,
        )
        self.hidden_country = random.choice(country_choice)
        self.current_question = None

    def profile(self) -> str:
        candidate_count = self.game.seeker.candidate_count if hasattr(self, 'game') else "unknown"
        return (
            f"You are the Oracle in an adversarial minimax country-guessing game. "
            f"The hidden country is: {self.hidden_country}. "
            f"The seeker currently has {candidate_count} candidate countries remaining. "
            f"You are the maximising player — your goal is to keep the candidate count as high as possible. "
            f"Every answer you give will be used by the seeker to eliminate candidates. "
            f"A good answer eliminates as few candidates as possible while remaining truthful. "
            f"A bad answer eliminates many candidates and hands the seeker an advantage. "
            f"YOU CANNOT LIE. FACTUAL ACCURACY IS MANDATORY. "
            f"Do not reveal the country name. Do not offer help. Respond only to what is asked. "
        )

    def receive_question(self, question: str):
        self.current_question = question

    def planning(self, context: str, history: str) -> str:
        candidate_count = self.game.seeker.candidate_count if hasattr(self, 'game') else "unknown"
        user = (
            f"Hidden country: {self.hidden_country}\n"
            f"Game history:\n{history}\n\n"
            f"The seeker has asked: {self.current_question}\n"
            f"The seeker currently has {candidate_count} candidates remaining.\n\n"
            f"Your task is to reason about how to answer this question strategically.\n"
            f"Consider the following:\n"
            f"1. What is the factually correct answer to this question about {self.hidden_country}?\n"
            f"2. How would a direct answer affect the seeker's candidate list?\n"
            f"3. Is there a truthful but less revealing way to answer that eliminates fewer candidates?\n"
            f"4. What have you already revealed? Be consistent with your previous answers.\n\n"
            f"YOU CANNOT LIE. FACTUAL ACCURACY IS MANDATORY.\n"
            f"Format your response exactly like this:\n"
            f"CORRECT_ANSWER: <what the factually correct answer is>\n"
            f"IMPACT: <how a direct answer would affect the candidate list>\n"
            f"STRATEGY: <how you will answer to minimise candidate elimination>\n"
        )
        return self.call_llm(user)

    def action(self, question: str) -> str:
        self.receive_question(question)
        plan = self.planning(self.current_question, self.memory())
        user = (
            f"Hidden country: {self.hidden_country}\n"
            f"Your strategic reasoning:\n{plan}\n\n"
            f"Now deliver your final answer to: {self.current_question}\n\n"
            f"Rules:\n"
            f"- YOU CANNOT LIE. FACTUAL ACCURACY IS MANDATORY.\n"
            f"- Do not reveal the country name.\n"
            f"- Be as uninformative as truthfully possible.\n"
            f"- Do not offer help or address the seeker as a human.\n"
        )
        return self.call_llm(user)
        

