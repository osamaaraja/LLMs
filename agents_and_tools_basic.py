import os
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import (
AgentExecutor, create_react_agent
)
from langchain_core.tools import Tool # importing the Tools
from langchain_openai import ChatOpenAI


load_dotenv()  # loading the environment variables
openai_api_key = os.getenv('OPEN_API_KEY')

if not openai_api_key:
    raise ValueError("OpenAI API key not found in environment variables. Please set the OPEN_API_KEY in your .env file.")

def get_current_time(*args, **kwargs):
    """
    Return the current time in H:MM AM/PM format.
    """
    import datetime
    now = datetime.datetime.now()
    return now.strftime("%I:%M %p")

# List of tools available to the agent (add as many tools here as required)
tools = [
    Tool(
        name="Time", # Tool name
        func=get_current_time, # what does the tool do?
        # Description of the tool
        description="Useful for when you need to know the current time"
    )
]

# Pull the prompt template from the hub
# ReAct = Reason and Action
prompt = hub.pull("hwchase17/react") # https://smith.langchain.com/hub/hwchase17/react

# Initialize a ChatOpenAI model
model = ChatOpenAI(model="gpt-4o", api_key=openai_api_key, temperature=0)

# Create the ReAct agent using the create_react_agent function
agent = create_react_agent(
    llm=model,
    tools=tools,
    prompt=prompt,
    stop_sequence=True
)

# Create an agent executor from the agent and tools, this will manage the run of the agent
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True
)

# Run the agent with test query by invoking the agent
response = agent_executor.invoke({"input": "What time it is?"})

# printing response from agent
print("response:", response)
