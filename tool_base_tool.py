import os
from dotenv import load_dotenv
from typing import Type
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()
openai_api_key = os.getenv('OPEN_API_KEY')

if not openai_api_key:
    raise ValueError("OpenAI API key not found in environment variables. Please set the OPEN_API_KEY in your .env file.")

class SimpleSearchInput(BaseModel):
    query: str = Field(description="should be a search query")

class MultiplyNumberArgs(BaseModel):
    x: float = Field(description="first number to multiply")
    y: float = Field(description="second number to multiply")

# custom tool with only custom input
# Inheriting from BaseTool to create new tools
class SimpleSearchTool(BaseTool):
    name = "simple_search"
    description = "useful when answering questions about current events"
    args_schema: Type[BaseModel] = SimpleSearchInput

    def _run(self, query: str) -> str:
        """Use the tool"""
        from tavily import TavilyClient # Connect your LLM to the web
        api_key = os.getenv('TAVILY_API_KEY')
        Client = TavilyClient(api_key)
        result = Client.search(query)
        return f"Search results for: {query}\n\n\n{result}\n"

# custom tool with custom input and output
class MultiplyNumbersTool(BaseTool):
    name = "multiply_numbers"
    description = "useful for multiplying two numbers"
    args_schema: Type[BaseModel] = MultiplyNumberArgs

    def _run(self, x:float, y:float) -> str:
        """Use the tool"""
        result = x*y
        return f"The product of {x} and {y} is {result}"

# create tools using Pydantic subclass approach
tools = [SimpleSearchTool(),MultiplyNumbersTool()]

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
response = agent_executor.invoke({"input": "Search for 'Apple Intelligence"})
print("Response for 'Apple Intelligence': ", response["output"])

response = agent_executor.invoke({"input": "Multiply 10 and 20"})
print("Response for 'Multiply 10 and 20': ", response["output"])