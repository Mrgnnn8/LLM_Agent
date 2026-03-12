from bot import Seeker, Oracle
import re

class GameEnvironment:
    def __init__(self, seeker: Seeker, oracle: Oracle):
        self.seeker = seeker
        self.oracle = oracle
        self.score = (self.seeker.candidate_count / len(self.seeker.country_choice)) * 100
        self.question_budget = seeker.question_budget
        self.game_over = False
        self.guess = None
        self.correct = None
        self.question = None

    def run(self):
        turn = 1
        print(f"The chosen country: {self.oracle.hidden_country}. THIS IS HIDDEN FROM THE SEEKER. ")
        print(f"Oracle: I've chosen my country, ask your first question...\n")
        print(f"The question budget for this round is {self.seeker.question_budget}")

        #print("=== SEEKER PROFILE ===")
        #print(self.seeker.profile())
        #print("=== ORACLE PROFILE ===")
        #print(self.oracle.profile())

        while self.seeker.questions_asked < self.question_budget:
            self.question = self.seeker.act()
            
            self.log_candidates(turn, self.seeker.last_plan)
            
            if self.question is None:
                break

            print(f"Seeker: {self.question}")
            answer = self.oracle.action(self.question)
            print(f"Oracle: {answer}")

            self.seeker.update_history(self.question, answer)
            self.oracle.update_history(self.question, answer)
            #self.score -= 10
            turn += 1

        self.guess = self.seeker.make_guess()
        self.correct = self.guess.lower() == self.oracle.hidden_country.lower()
        
        if self.correct:
            winner = "Seeker"
            self.game_over = True
        else:
            winner = "Oracle"
            self.game_over = True

        print(f"\nThe Seeker guessed {self.guess}")
        print(f"The seeker used {self.seeker.questions_remaining} / {self.seeker.question_budget} questions. ")
        print(f"\n{winner} wins! The correct answer was {self.oracle.hidden_country}")

        #print(f"\nSeeker's final guess: {self.guess}")
        #print(f"Correct answer: {self.oracle.hidden_country}")
        #print(f"Correct: {self.correct}")
        #print(f"Score: {self.score}")

    def result(self) -> dict:
        if not self.game_over:
            return None
        return {
            "guess": self.guess,
            "correct_answer": self.oracle.hidden_country,
            "correct": self.correct,
            "question_asked": self.seeker.questions_asked,
            "score": self.score
        }

    def log_candidates(self, turn: int, plan: str):
        #print(f"RAW PLAN:\n{plan}\n") #TEMPORARY
        try: 
            candidates_line = [line for line in plan.split ("\n") if line.startswith("CANDIDATES:") ][0]
            candidates = candidates_line.replace("CANDIDATES:", "").strip()
        except IndexError:
            candidates = "Could not parse candidates"

        code_count = self.seeker.candidate_count
        
        with open("candidate_log.txt", "a") as f:
            f.write(f"Turn {turn} | Running Score: {code_count} | The last question: {self.question} | Candidates: {candidates}\n")
