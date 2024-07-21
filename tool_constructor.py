from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.tools import StructuredTool, Tool
from langchain_openai import OpenAI, OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
openai_api_key = os.getenv('OPEN_API_KEY')

if not openai_api_key:
    raise ValueError("OpenAI API key not found in environment variables. Please set the OPEN_API_KEY in your .env file.")
# Functions for the tools
def greet_user(name: str) ->str:
    """Greets the user by name"""
    return f"Hello {name}!"

def reverse_string(text: str) -> str:
    """Reverses the given string"""
    return text[::-1]

def concatenate_strings(a: str, b: str) -> str:
    """Concatenation of two strings"""
    return a + b

# Pydantic model for tool arguments
class ConcatenateStringsArgs(BaseModel):
    a: str = Field(description="First string")
    b: str = Field(description="Second string")

# create tools using the Tool and StructuredTool constructor approach
tools =[
    Tool(
        name="GreetUser",
        func=greet_user,
        description="Greet the user by name"
    ),

    Tool(
        name="ReverseString",
        func=reverse_string,
        description="Reverse the given string"
    ),

    # Use StructuredTool for more complex functions that require multiple input parameters
    # StructuredTool allows us to define an input schema using Pydantic
    # any function that takes in more than one parameter, better to go for StructuredTool
    StructuredTool.from_function(
        func=concatenate_strings,
        name="ConcatenateStrings",
        description="Concatenate two strings",
        args_schema=ConcatenateStringsArgs
    )
]

# Initialize a ChatOpenAI model
model = ChatOpenAI(model="gpt-4o", api_key=openai_api_key, temperature=0)

# pulling the prompt template form hub
prompt = hub.pull("hwchase17/openai-tools-agent")

agent = create_tool_calling_agent(
    llm=model,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

# Test the agent with sample queries
response = agent_executor.invoke({"input": "Greet Groot"})
print("Response for 'Greet Groot': ", response["output"])

response = agent_executor.invoke({"input": "Reverse the string 'hello'"})
print("Response for Reverse the string 'hello': ", response["output"])

response = agent_executor.invoke({"input": "Concatenate 'hello' and 'world'"})
print("Response for Concatenate 'hello' and 'world': ", response["output"])
