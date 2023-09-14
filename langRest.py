from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.chains import SequentialChain
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.llms import OpenAI

import os
os.environ['OPENAI_API_KEY'] = "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"

llm = OpenAI(temperature=0.9)
name = llm.predict("I want to open a restaurant for Indian food. Suggest a fancy name for this.")
print(name)

llm("I want to open a restaurant for Indian food. Suggest a fancy name for this.")

prompt_template_name = PromptTemplate(
    input_variables=['cuisine'],
    template="I want to open a restaurant for {cuisine} food. Suggest a fancy name for this."
)
p = prompt_template_name.format(cuisine="Italian")
print(p)


chain = LLMChain(llm=llm, prompt=prompt_template_name)
chain.run("Mexican")


chain = LLMChain(llm=llm, prompt=prompt_template_name, verbose=True)
chain.run("Mexican")

llm = OpenAI(temperature=0.6)

prompt_template_name = PromptTemplate(
    input_variables=['cuisine'],
    template="I want to open a restaurant for {cuisine} food. Suggest a fancy name for this."
)

name_chain = LLMChain(llm=llm, prompt=prompt_template_name)

prompt_template_items = PromptTemplate(
    input_variables=['restaurant_name'],
    template="""Suggest some menu items for {restaurant_name}"""
)
food_items_chain = LLMChain(llm=llm, prompt=prompt_template_items)


chain = SimpleSequentialChain(chains=[name_chain, food_items_chain])

content = chain.run("Indian")
print(content)

llm = OpenAI(temperature=0.7)

prompt_template_name = PromptTemplate(
    input_variables=['cuisine'],
    template="I want to open a restaurant for {cuisine} food. Suggest a fency name for this."
)

name_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key="restaurant_name")

llm = OpenAI(temperature=0.7)

prompt_template_items = PromptTemplate(
    input_variables=['restaurant_name'],
    template="Suggest some menu items for {restaurant_name}."
)

food_items_chain = LLMChain(llm=llm, prompt=prompt_template_items, output_key="menu_items")
chain = SequentialChain(
    chains=[name_chain, food_items_chain],
    input_variables=['cuisine'],
    output_variables=['restaurant_name', "menu_items"]
)

chain({"cuisine": "Indian"})

os.environ['SERPAPI_API_KEY'] = "010c2097255a74b85c24cc859af367552bf529e4284a96583e7c4f5c952be788"


llm = OpenAI(temperature=0)

# The tools we'll give the Agent access to. Note that the 'llm-math' tool uses an LLM, so we need to pass that in.
tools = load_tools(["serpapi", "llm-math"], llm=llm)

# Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Let's test it out!
agent.run("What was the GDP of US in 2022 plus 5?")

# The tools we'll give the Agent access to. Note that the 'llm-math' tool uses an LLM, so we need to pass that in.
tools = load_tools(["wikipedia", "llm-math"], llm=llm)

# Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Let's test it out!
agent.run("When was Elon musk born? What is his age right now in 2023?")
chain = LLMChain(llm=llm, prompt=prompt_template_name)
name = chain.run("Mexican")
print(name)
name = chain.run("Arabic")
print(name)
print(chain.memory.buffer)
