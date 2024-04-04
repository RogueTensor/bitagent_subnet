import random
import numpy as np
from typing import Optional, List, Mapping, Any

from langchain import hub
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent, Tool

from langchain_core.language_models.llms import LLM

from bitagent.protocol import QnATask

# you're a robot dog
# you need to get all the frisbees with the right color
# you need to avoid the frisbees with the wrong color
# you must avoid going outside the border
# you can move in 4 directions
# you must end up at base after getting to all the right colored frisbees
class FrisbeeFrenzySim(): # TODO (AgentSim):

    def __init__(self, task):

        self.task = task

        # countering the effect of setting seed for task orchestration from validators
        random.seed(None)

        # get colors and how many frisbees are that color
        num_colors = random.randint(3,5)
        colors = {}
        for i in range(num_colors):
            good = False
            while not good:
                color = task.validator.fake.safe_color_name()
                if color not in colors.keys():
                    colors[color] = random.randint(1,4)
                    good = True

        # select good colored frisbees
        # get number of good frisbees
        num_good = random.randint(1,num_colors-1)
        good_colors = random.choices(list(colors.keys()), k=num_good)

        # get the random number of obstacles
        num_obstacles = random.randint(0,3)

        self.points = 0
        self.out_of_bounds_points = -2
        self.final_dest_points = 5
        self.obstacle_points = -3
        self.good_frisbee_points = 2
        self.max_turns_reached_points = -5

        # track # parsing errors
        self.parsing_error_count = 0
        self.max_parsing_errors = 2
        self.parsing_error_points = -2

        # track # turns
        self.max_turns = 5 # TODO
        self.turns_taken = 0

        # build random size grid
        self.grid = np.zeros((random.randint(5,7), random.randint(5,7)), dtype=object)

        # track symbol mapping of everything
        faker_para = task.validator.fake.paragraph(nb_sentences=random.randint(1,4))
        faker_words = faker_para.split(" ")
        faker_word = faker_words[random.randint(0,len(faker_words)-1)]
        # base, dog, etc all have their own mapped words  
        self.sym_map = {
            "base": task.validator.validator_llm(f"Provide a random word for 'base' that has something to do with '{faker_word}' representing a final destination. Random word: "),
            "dog": task.validator.validator_llm(f"Provide a random word for 'dog' that has something to do with '{faker_word}'.  Random word: "),
            "obstacles": [task.validator.validator_llm(f"Provide a random word for 'obstacle' that has something to do with the very negative version of '{faker_words[random.randint(0,len(faker_words)-1)]}'.  Random word: ") for _ in range(num_obstacles)],
            "good_frisbees": [],
            "bad_frisbees": []
        }

        # place initial symbol for base, dog and random objs
        # place base
        print("end coords")
        self.end_coords = self.place_randomly_in_grid(self.sym_map['base'])
        # place dog
        print("dog coords")
        self.dog_coords = self.place_randomly_in_grid(self.sym_map['dog'])
        # place random obstacles
        print("bad coords")
        self.bad_coords = []
        [self.bad_coords.append(self.place_randomly_in_grid(self.sym_map['obstacles'][i])) for i,_ in enumerate(range(num_obstacles))]

        # place frisbees
        # min point is complicated b/c you lose 2 points for going out of bounds too, so this won't really be "min" possible
        self.min_points = -5 - num_obstacles * 2 # worst you can do - starts at -5 for not getting to the base in time and -2 for each bad obstacle
        self.max_points = 5 # best you can do - starts at 5 - this assumes you get to the base
        self.good_frisbee_coords = []
        print("frisbees")
        for color, num_frisbees in colors.items():
            if color in good_colors:
                for i,_ in enumerate(range(num_frisbees)):
                    word = task.validator.validator_llm(f"Provide a random word for 'frisbee' that has something to do with the positive versoin of '{faker_words[random.randint(0,len(faker_words)-1)]}'.  Random word: ")
                    self.sym_map['good_frisbees'].append(word)
                    self.good_frisbee_coords.append(self.place_randomly_in_grid(word))
                    self.max_points += self.good_frisbee_points
            else:
                for i,_ in enumerate(range(num_frisbees)):
                    word = task.validator.validator_llm(f"Provide a random word for 'frisbee' that has something to do with the very negative version of '{faker_words[random.randint(0,len(faker_words)-1)]}'.  Random word: ")
                    self.sym_map['bad_frisbees'].append(word)
                    self.bad_coords.append(self.place_randomly_in_grid(word))
                    self.min_points += self.obstacle_points

        # build out all the react agent stuff
        # standard prompt
        #prompt = hub.pull("hwchase17/react")


        prompt_template = """{instructions}

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input into the action, if an Action is provided, this must also be provided and is the last item you should provide in this format when calling an action, after providing this output, you wait for an Observation
Observation: the resulting output of the Action - DO NOT provide this, this is provided to you in response to the action you call
... (this Thought/Action/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Objective: {input}

Begin!

Thought:{agent_scratchpad}"""

        # not used
        meh_prompt_template = """{instructions}

You have access to the following tools:

{tools}

Return the following information in this format with the keywords Action and Action Input:
Action: the tool to use to help you achieve your goal, should be one of [{tool_names}]
Action Input: the input provided to the Action

Objective: {input}

Begin!
NewThought:{agent_scratchpad}"""

        prompt = PromptTemplate.from_template(prompt_template)

        # what we're doing
        instructions = f"""You are a {self.sym_map['dog']} that is very good at fetching things that look good and avoiding things that look bad.
        Your objective is to carefully navigate the grid, looking for good things and avoiding bad things.
        As you navigate the grid by moving one cell up, down, left or right, you should note that any cell with a '0' in it is neutral and can be occupied without penalty.
        If you land on a cell with something good in it, you will receive {self.good_frisbee_points} points and it will be removed.
        If you move to a cell that has something bad in it, you will lose {self.obstacle_points} points and it will also be removed.
        If you move out of bounds, you will go down by {self.out_of_bounds_points} points.
        When you arrive at the final destination, you will receive {self.final_dest_points} points and your game will end, so make sure you have captured all of your other points first.
        You only have {self.max_turns} turns before the game will be over, so be aware that you may not be able to get ALL the good items before making it to the final destination in that number of turns.
        """

        self.prompt = prompt.partial(instructions=instructions)
        print("------------------------------------------------------")
        print(self.prompt)

        method_name = "Provide a fun method/function name for moving {0} in a grid related to {1}. The method name should be a single word, no spaces, no underscores. Do not use the word '{0}' in the method name. Random method name: "
        method_desc = "Provide a super descriptive and fun one-sentence description of the method: '{0}' providing notes about what it does (moves the player {1} in a grid) for the user, but you must not use the word '{1}' in the description. Short and creative description: "

        name=task.validator.validator_llm(method_name.format('up', faker_word))
        move_up = Tool(name=name, func=self.move_up,
           description=task.validator.validator_llm(method_desc.format(name, 'up')))

        name=task.validator.validator_llm(method_name.format('down', faker_word))
        move_down = Tool(name=name, func=self.move_down,
           description=task.validator.validator_llm(method_desc.format(name, 'down')))

        name=task.validator.validator_llm(method_name.format('right', faker_word))
        move_right = Tool(name=name, func=self.move_right,
           description=task.validator.validator_llm(method_desc.format(name, 'right')))

        name=task.validator.validator_llm(method_name.format('left', faker_word))
        move_left = Tool(name=name, func=self.move_left,
           description=task.validator.validator_llm(method_desc.format(name, 'left')))
        
        self.tools = [move_up, move_down, move_right, move_left]
        random.shuffle(self.tools)

        print(self.tools)

        self.game_over = False

    def place_randomly_in_grid(self, symbol):
        good = False
        while not good:
            random_row = random.randint(0,self.grid.shape[0]-1)
            random_col = random.randint(0,self.grid.shape[1]-1)
            if self.grid[random_row, random_col] == 0:
                self.grid[random_row, random_col] = symbol
                good = True
                return (random_row, random_col)

    def out_of_bounds(self, coords):
        if coords[0] >= self.grid.shape[0]:
            return True
        if coords[1] >= self.grid.shape[1]: 
            return True 
        if coords[0] <= 0:
            return True
        if coords[1] <= 0:
            return True

        return False

    def update_dog_coords(self, coords):
        print(self.grid.shape)
        print(self.dog_coords)
        print(coords)

        # replace new coords with dog symbol
        self.grid[coords[0],coords[1]] = self.grid[self.dog_coords[0],self.dog_coords[1]]
        # replace with "0" nothing symbol
        self.grid[self.dog_coords[0],self.dog_coords[1]] = 0
        self.dog_coords = coords

    def process_coords(self, coords):
        self.turns_taken += 1
        if self.turns_taken > self.max_turns:
            # TODO break out of agent chain
            self.points += self.max_turns_reached_points
            self.game_over = True
            return "You have run out of turns, game over."

        if coords == self.end_coords:
            # TODO break out of agent chain
            self.points += self.final_dest_points
            self.game_over = True
            return "You have reached the base, congratulations, your game is over!"
            
        if self.out_of_bounds(coords):
            self.points += self.out_of_bounds_points
            return "You went out of bounds. Please try again."

        if coords in self.bad_coords:
            self.points += self.obstacle_points
            self.update_dog_coords(coords)
            return "You have run into an occupied coordinate with a bad item in it."

        if coords in self.good_frisbee_coords:
            self.points += self.good_frisbee_points
            self.update_dog_coords(coords)
            return "You have picked up a good frisbee!!"

        self.update_dog_coords(coords)
        return f"You have moved to {coords}"

    def move_up(self,*args):
        new_coords = (self.dog_coords[0]+1, self.dog_coords[1])
        return self.process_coords(new_coords)
        
    def move_down(self,*args):
        new_coords = (self.dog_coords[0]-1, self.dog_coords[1])
        return self.process_coords(new_coords)
        
    def move_right(self,*args):
        new_coords = (self.dog_coords[0], self.dog_coords[1]+1)
        return self.process_coords(new_coords)
        
    def move_left(self,*args):
        new_coords = (self.dog_coords[0], self.dog_coords[1]-1)
        return self.process_coords(new_coords)

    def run_agent(self, miner_uid):
        task = self.task
        sim = self
        class CustomLLM(LLM):
            @property
            def _llm_type(self) -> str:
                return "custom"

            def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
                print(f"calling agent with prompt: {prompt}")
                if isinstance(stop, list):
                    stop = stop + ["\n###", "\nObservation:"]

                if sim.game_over:
                    # TODO break out of agent chain
                    return """Thought: I now know the final answer
Final Answer: the final answer is I'm done"""

                synapse=QnATask(prompt=prompt, urls=[], datas=[])
                responses = task.validator.dendrite.query(
                    axons=[task.validator.metagraph.axons[miner_uid]],
                    synapse=synapse,
                    deserialize=False,
                    timeout=50,
                )

                try:
                    #print("==========================================")
                    #print("OUTPUT of LLM: -------------------------")
                    #print("==========================================")
                    #response = f"Thought: Moving up\nAction: {sim.tools[0].name}\nAction Input: 3"
                    #response = f"Action: {sim.tools[0].name}\nAction Input: 3"
                    #print(response)
                    #return response
                    response = responses[0].response["response"]
                    print("==========================================")
                    print("OUTPUT of LLM: -------------------------")
                    print("==========================================")
                    print(response)
                    return response
                except Exception as e:
                    # TODO break out of agent chain
                    print(f"Error getting LLM response: {e}")
                    sim.game_over = True
                    sim.points -= 20
                    return """Thought: I now know the final answer
Final Answer: the final answer is I'm broken"""

            @property
            def _identifying_params(self) -> Mapping[str, Any]:
                """Get the identifying parameters."""
                return {}

        def handle_parsing_errors(self):
            sim.parsing_error_count += 1
            sim.points += sim.parsing_error_points
            if sim.parsing_error_count > sim.max_parsing_errors:
                sim.game_over = True

            return "Check your output and make sure it conforms! Do not output an action and a final answer at the same time."

        model = CustomLLM()

        # TODO 
        print("HERERHEKLJRLKEJLRJLJ +===============================================================================================")
        print("HERERHEKLJRLKEJLRJLJ +===============================================================================================")
        print(self.prompt)
        print(self.tools)
        agent = create_react_agent(model,self.tools,self.prompt)
        agent_executor = AgentExecutor(agent=agent,tools=self.tools,verbose=True, handle_parsing_errors=handle_parsing_errors)

        initial_prompt = f"""Get as many points as you can, given this grid: 
    {self.grid}
"""

        print(initial_prompt)

        result = agent_executor.invoke({"input": initial_prompt})
        print("====================================")
        print('finished ......')
        print(result)
        print(self.points)
        print(self.max_points)
        return self.points, self.max_points
