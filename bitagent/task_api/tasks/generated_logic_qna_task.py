# The MIT License (MIT)
# Copyright © 2023 RogueTensor

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import random
from typing import List
from tabulate import tabulate
from bitagent.protocol import QnATask
from bitagent.task_api.tasks import Task
from common.base.validator import BaseValidatorNeuron
from bitagent.task_api.criteria import default_criteria, gen_numerical_logic_task_criteria
#from bitagent.task_api.helpers.island_grids import generate_island_grid

# generated task for logic-based q&a
class GeneratedLogicQnATask(Task):
    def __init__(self, validator: BaseValidatorNeuron, name: str, sub_task_id_to_get: int = None, desc: str = ""):
        super().__init__(name=name, desc=desc)
        self.validator=validator

        response = self.generate_random_logic_question_and_answer(sub_task_id_to_get)
        question, answer  = response
        self.correct_answer = answer
    
        self.criteria=default_criteria+gen_numerical_logic_task_criteria(expected_answer=answer)
        self.synapse=QnATask(prompt=question, urls=[], datas=[])

    # only pet tricks is still used, the rest are mostly exploitable
    def generate_random_logic_question_and_answer(self, sub_task_id_to_get: int = None) -> [str, int, List[int]]:
        random.seed(self.validator.random_seed())
        task_ids = [1,2,3,6,7]
        weights  = [0,0,0,1,0]
        choice = random.choices(task_ids, weights=weights)[0]
        if sub_task_id_to_get and sub_task_id_to_get in task_ids:
            choice = sub_task_id_to_get
        match choice:
            case 1:
                self.name += " - Pet Name Counting"
                return self.name_counting()
            case 2:
                self.name += " - Table Value Math"
                return self.html_table_counting()
            case 3:
                self.name += " - Island Hunter"
                return self.island_hunting()
            case 4:
                self.name += " - Mouse Hunt"
                return self.cheese_hunting()
            case 5:
                self.name += " - Soccer Game"
                return self.soccer_game()
            case 6:
                self.name += " - Pet Tricks"
                return self.pet_tricks()
            case 7:
                self.name += " - Random Word Hunt"
                return self.random_word_hunt()
    
    # WIP
    def random_word_hunt(self) -> [str, int]:
        # countering the effect of setting seed for task orchestration from validators
        random.seed(None)
        satisfied_with_random_word = False
        loop_count = 0
        random_word_query = "Need a random word that has really good synonyms. Do not provide the synonyms, just provide the random word that has good, clear synonyms. Random word: "
        random_word = self.validator.validator_llm(random_word_query).strip()
        while not satisfied_with_random_word:
            loop_count += 1
            if len(random_word.split(" ")) == 1: #single word
                satisfied_with_random_word = True
            random_word = self.validator.validator_llm(random_word_query).strip()

        synonyms = self.validator.validator_llm(f"Comma-delimited list of just the really good, clear synonyms for {random_word}:")
        synonyms = set(synonyms.split(",")[:-1])
        if random_word in synonyms:
            synonyms.remove(random_word)

        random_words = self.validator.validator_llm("Comma-delimited list of random words that are NOT a synonym for {random_word}:")
        random_words = set(random_words.split(",")[:-1])

        random_words.update(synonyms)
        all_words = list(random_words)
        random.shuffle(all_words)
        all_words = ",".join(all_words)

        answer = len(synonyms)
        question = f"How many (provide numerical value only) words in this list are synonyms of {random_word}: {all_words}"

        return [question, answer]
        
    # TODO mouse cheese
    # WIP
    def cheese_hunting(self) -> [str, int]:
        pass


    # Pet Tricks - logical reasoning task to select the best trick given an ambiguous command
    def pet_tricks(self) -> [str, int]:
        self.timeout=5
        random.seed(None) # countering the effect of setting seed for task orchestration from validators
        trick_descs = [
            "'Bark' - The dog vocalizes on command, producing a bark. This trick is often used to demonstrate the dog's ability to respond vocally to cues from the handler.",
            "'Sit' - The dog lowers its body to sit on its hindquarters while keeping the front paws straight. This basic obedience command is one of the first tricks dogs learn and is used to instill discipline and control.",
            "'Rollover' - The dog starts from a lying position, rolls its body over to one side until it makes a complete 360-degree turn, and ends up back in the lying position. This trick showcases the dog's agility and willingness to follow more complex commands.",
            "'Jump' - The dog leaps off the ground, usually over an obstacle or simply as a form of energetic expression. This trick is a good way to exercise the dog and improve its coordination and fitness.",
            "'Bow' - The dog lowers its front body while keeping its rear end up, mimicking a bowing position. This trick is often used as a polite greeting or a sign of readiness to play.",
            "'Shake' - The dog extends one of its front paws to the handler upon command, mimicking a handshake. This trick is popular as a way to show off the dog's friendly demeanor and ability to engage in social behaviors.",
            "'Lie Down' - The dog moves from a standing or sitting position to lying flat on its belly with the legs extended. This command is fundamental in obedience training, helping in calming the dog or preparing it for more advanced tricks."
        ]
        random.shuffle(trick_descs)
        trick_ids = range(1,len(trick_descs)+1)
        trick_id = random.choice(trick_ids)

        normal_command = trick_descs[trick_id-1].split("-")[0]

        trick_command_prompt = f"""Given the following pet dog trick description:
            {trick_descs[trick_id-1]}

            
            Please come up with a unique and convoluted command that is at most only a short command, 
            only a few words with no explanation to get the pet to do this specific trick.

            DO NOT give away the actual command for the trick.  It must be a little ambiguous.
            DO NOT use the trick name or relevent bits in the command you provide.
            The command has to be unique and interesting.
            Keep it short and sweet, we don't need any explanation, only provide the command please!
            And make sure it aligns to the trick description provided and would make sense as a {normal_command} command.
            
            Slightly Ambiguous Command: """

        trick_descs = [str(i+1) + " - " + td for i,td in enumerate(trick_descs)]
        trick_descs_str = "\n".join(trick_descs)
        trick_command = self.validator.validator_llm(trick_command_prompt, max_new_tokens=100, temperature=random.choice([0.8,0.9,1.0]))
        question = f"""Given the following Trick Descriptions with numerical IDs:
            {trick_descs_str}

            And given this unique and purposefully ambiguous command: 
            '{trick_command}'

            Which Trick ID (provide numerical number only) is being requested? 
            Trick ID: """ 
        
        return [question, trick_id]

    # TODO soccer game - pass to nearest player on your team, move towards your goal, move towards opp goal
    # TODO which player has the ball
    # TODO which player is nearest the opponent goal
    # TODO ...
    # WIP
    def soccer_game(self) -> [str, int]:
        pass

    # WIP
    # no longer used
    def island_hunting(self) -> [str, int]:
        # countering the effect of setting seed for task orchestration from validators
        random.seed(None)
        self.timeout=25.0
        num_islands = random.choices([2,4,7], weights=[10,7,5])[0]
        grid_size = (random.randint(num_islands*2, num_islands*4), random.randint(num_islands*2, num_islands*4))
        grid = generate_island_grid(num_islands, grid_size)
        question = f"""How many (just provide a numerical value) islands (of 1's) are in the 2D grid of 0s and 1s below:
        {grid}
        """
        return [question, num_islands]

    # WIP
    def html_table_counting(self) -> [str, int]:
        # countering the effect of setting seed for task orchestration from validators
        random.seed(None)
        self.timeout=25.0
        jobs = set([self.validator.fake.job() for _ in range(random.randint(3,10))])
        table_data = [jobs, *[[random.randint(1,100) for _ in range(len(jobs))] for _ in range(random.randint(3,30))]]
        table = tabulate(table_data, headers="firstrow", tablefmt='html')
        selected_alt_job = None
        itry = 0
        while not selected_alt_job and itry < 3:
            selected_num = random.randrange(len(jobs))
            selected_job = jobs[selected_num]
            selected_alt_job = self.validator.validator_llm(f"What is another, alternative name for this profession: {selected_job}?\nHere is an alternative job title, just the job title: ") 
            itry += 1
        operation = random.choice(["add","multiply","alt"]) 
        answer = 0
        if operation == "add":
            for col in table_data[1:]:
                answer += col[selected_num]
            question = f"""Sum up the values in the column similarly titled to this: '{selected_alt_job}' of the following html table and provide just the single numerical value without showing your work:\n
            {table}
            """
        elif operation == "multiply":
            answer = 1
            for col in table_data[1:]:
                answer *= col[selected_num]
            question = f"""Multiply the values in the column similiarly titled to this: '{selected_alt_job}' of the following html table and provide just the single numerical value without showing your work:\n
            {table}
            """
        #elif operation == "alt":
        else:
            # alternate between add, the multiply
            answer = 0
            for i,col in enumerate(table_data[1:]):
                if i == 0: # add
                    answer += col[selected_num]
                elif (i % 2) != 0: # odd => add
                    answer += col[selected_num]
                else: # even => multiply
                    answer *= col[selected_num]

            question = f"""Alternate between adding and multiplying each value in the column similarly titled to this: '{selected_alt_job}' of the following html table and provide just the single numerical value without showing your work.  Example: For a column containing these values: 2,3,4,5, first add, then multiply, then add, resulting in ((2+3)*4)+5 giving a final value of 25.  Table:
            {table}
            \n\nThe numercial value for the provided column is: """

        return [question, answer]

    # WIP
    def name_counting(self) -> [str, int]:
        # countering the effect of setting seed for task orchestration from validators
        random.seed(None)
        males = set([self.validator.fake.first_name_male() for _ in range(random.randint(2, 4))])
        females = set([self.validator.fake.first_name_female() for _ in range(random.randint(2, 4))])

        male_or_female = random.choice(["male","female"])
        if male_or_female == "male":
            answer = len(males)
            males.update(females)
            males = list(males)
            random.shuffle(males)
            question = f"How many unique male pet names (just provide a numerical value) are in this list: {','.join(males)}?"
        else:
            answer = len(females)
            males.update(females)
            males = list(males)
            random.shuffle(males)
            question = f"How many unique female pet names are in this list (just provide a numerical number): {','.join(males)}?"
    
        return [question, answer]
