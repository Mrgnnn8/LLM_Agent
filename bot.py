import os
import json
from openai import OpenAI
from abc import ABC, abstractmethod


class Brain(ABC):
    def __init__(self, client: str, role: str, question_budget: int, model: str, attribute_space: list):
        self.api_client = client
        self.model = model

        self.role = role
        self.history = []
        self.max_history = 20
        self.question_budget = question_budget
        self.questions_asked = 0
        self.attribute_space = attribute_space

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
        Retrieves and formats he conversation history for the injection into the prompt.
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

    def prompt_constructor(self) -> str:
        """
        Orchestrates the four modules in sequence to build a prompt which is fed into the model.
        """
        context = self.profile()
        history = self.memory()
        plan = self.plan(context, history)
        output = self.action(plan)
        return output
    
    def call_llm(self, input: str, user: str) -> str:
        response = self.api_client.responses.create(
            model=self.model,
            instructions=self.profile(),
            input=input
        )
        return response.output_text.strip()

    def update_history(self, question: str, answer: str): 
        self.history.append({"question": question, "answer": answer})


class Seeker(Brain):
    def __init__(self, model: str, question_budget: int, attribute_space: list):
        super().__init__(
            role="seeker",
            model=model, 
            question_budget=question_budget,
            attribute_space=attribute_space
            )

    def profile(self) -> str:
        budget_remaining = self.question_budget - self.questions_asked
        attributes = (
            f"You may only ask questions about the following attributes: {', '.join(self.attribute_space)}"
            if self.attribute_space else 
            "You may ask yes/no questions about the country."
        )
        return (
            f"You are a strategic question-asker trying to identify a hidden country. "
            f"Your goal is to identify the country in as few questions as possible. "
            f"{attributes} "
            f"You have {budget_remaining} questions remaining out of {self.question_budget}."
        )

    def planning(self, context: str, history: str) -> str:
        system = context
        user = (
            f"Game history so far:\n{history}\n\n"
            f"Before asking your next question, reason about your strategy. "
            f"Consider: What do you already know? What question would eliminate "
            f"the most candidate countries? Think step by step, then state your plan."
        )
        return self.call_llm(system, user)

    def action(self, plan: str) -> str:
        system = (
            "You are a question-asker in a country guessing game. "
            "Based on your reasoning, output only the next yes/no question to ask. "
            "DO not include any explanation - just the question itself. "
        )
        user = f"Your reasoning and plan:|n{plan}|n|nNow output your question:"
        question = self.call_llm(system, user)
        self.questions_asked += 1
        return question

    def make_guess(self) -> str:
        system = "You are trying to identify a hidden country based on the answers you have recieved. "
        user = (
            f"Game history:\n{self.memory}\n\n"  
            f"Based on everything you know, what is your final guess for the country? "
            f"Respond with only the country name."
        )
        return self.call_llm(system, user)



