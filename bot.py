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
        self.n_branches = 5 # controls number of thought branches

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

    def tree_of_thought(self, current_candidates: list, history: str) -> str:
        branches = []
        current_candidates = self.candidate_list()

        for i in range(1, self.n_branches + 1):
            #print(f"Thinking branch {i}/{self.n_branches}")
            user = (
                f"You are on thought {i} of {self.n_branches}.\n"
                f"previously asked questions, do not ask these again: {self.history}\n"
                f"you can ask questions from {self.attribute_space}"
                f"The number of candidate countries is: {self.candidate_count}.\n"
                f"The remaining candidate countries are {current_candidates}"
                f"Estimate how many candidates would remain for both a yes and no answer.\n"
                f"IMPORTANT: IF_YES_COUNT + IF_NO_COUNT must equal exactly {len(current_candidates)}.\n\n"
                f"QUESTION: <your question>\n"
                f"IF_YES_COUNT: <number>\n"
                f"IF_NO_COUNT: <number>\n"
            )
            response = self.call_llm(user)

            try:
                question = [l for l in response.split("\n") if l.startswith("QUESTION:")][0].replace("QUESTION:", "").strip()
                yes = int([l for l in response.split("\n") if l.startswith("IF_YES_COUNT:")][0].replace("IF_YES_COUNT:", "").strip())
                no = int([l for l in response.split("\n") if l.startswith("IF_NO_COUNT:")][0].replace("IF_NO_COUNT:", "").strip())
            except (IndexError, ValueError):
                question = "unknown"
                yes = no = len(current_candidates)

            branches.append({
                "branch_number": i,
                "score": self.game.score,
                "question": question,
                "if_yes_count": yes,
                "if_no_count": no,
            })
        
        #print(branches)
        self.log_branches(branches)
        self.branches = branches
        return branches

    def log_branches(self, branches: list):
        current_candidates = self.candidate_list()
        with open("tree_of_thoughts.txt", "a") as f:
            f.write(f"\nQuestion number: {self.questions_asked + 1} / {self.question_budget} |Branches evaluated: {len(branches)} | Candidates: {len(current_candidates)}\n")
            for b in branches:
                f.write(f"Branch {b['branch_number']}: {b['question']} | IF_YES_COUNT: {b['if_yes_count']} | IF_NO_COUNT: {b['if_no_count']}\n")

    def planning(self, context: str, history: str) -> str:
        current_candidates = self.candidate_list()
        branches = self.tree_of_thought(current_candidates, history)
        branches_summary = "\n".join([
            f"Option {b['branch_number']}: {b['question']} | If the answer is yes: {b['if_yes_count']} | If the reply is no: {b['if_no_count']}"
            for b in branches
        ])

        user = (
            f"Game history so far:\n{history}\n\n "
            f"The current score is {self.game.score}. You want to reduce the amount of candidates remaining as much as possible to help minimise the score. "
            f"You have already gone through and decided 5 questions you may ask, you must choose one of these questions: {branches_summary}"
            f"You have {self.question_budget - self.questions_asked} questions remaining.\n\n"
            f"Based on the current chat history, reason through the following steps:\n"
            f"1. What do you know so far about the country?\n"
            f"2. Given what you know, list the countries that are still possible from this list: {current_candidates}\n"
            f"3. How many candidates remain?\n"
            f"4. What is your strategy for your next question to eliminate the most candidates using the questions and outcomes in {branches_summary}?\n\n"
            f"Format your response exactly like this:\n"
            f"REASONING: <your reasoning>\n"
            f"CANDIDATES: <comma seperated list of countries which you believe the hidden country may be.>\n"
            f"STRATEGY: <your next question strategy which must stem from {branches}>\n"
        )
        plan = self.call_llm(user)
        #print(plan)
        return plan

    def action(self, plan: str) -> str:
        
        try:
            count_line = [line for line in plan.split("\n") if line.startswith("CANDIDATES:")][0]
            self.candidate_count = count_line.replace("CANDIDATES:", "").strip()
        except (IndexError, ValueError):
            self.candidate_count = len(self.country_choice)

        if len(self.candidate_count) <= 2:
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
        
        current_candidates = last_line.split(", ") if last_line else self.country_choice
        return current_candidates

    
    def update_candidate_file(self, question: str, answer: str):
        current_candidates = self.candidate_list()
    
        user = (
            f"Question asked: {self.game.question}\n"
            f"Oracle answered: {self.game.answer}\n\n"
            f"From this list:\n{', '.join(current_candidates)}\n\n"
            f"Remove any country that is inconsistent with the answer.\n"
            f"Label all remaining valid countries with CANDIDATE.\n"
            f"CANDIDATE: <country>\n"
        )
        response = self.call_llm(user)
    
        candidates = [
            line.replace("CANDIDATE:", "").strip()
            for line in response.split("\n")
            if line.startswith("CANDIDATE:")
        ]
        if not candidates:
            candidates = current_candidates
    
        with open("candidate_log.txt", "w") as f:
            f.write(", ".join(candidates))


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
        answer = self.call_llm(user)
        return answer

