import os
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()
openai_api_key = os.getenv('OPEN_API_KEY')

if not openai_api_key:
    raise ValueError("OpenAI API key not found in environment variables. Please set the OPEN_API_KEY in your .env file.")

def get_current_time(*args, **kwargs):
    """Return the current time in H:MM AM/PM format."""
    from datetime import datetime
    now = datetime.now()
    return now.strftime("%I:%M %p")

def search_wikipedia(query):
    """Search Wikipedia and return the summary of the first result."""
    from wikipedia import summary
    try:
        return summary(query, sentences=2)
    except Exception:
        return "I couldn't find any information on that."

# Define tools
tools = [
    Tool(
        name="Time",
        func=get_current_time,
        description="Useful for when you need to know the current time."
    ),
    Tool(
        name="Wikipedia",
        func=search_wikipedia,
        description="Useful for when you need to know about a topic."
    )
]

# Pull the prompt template from the hub
prompt = hub.pull("hwchase17/structured-chat-agent")

# Initialize ChatOpenAI model
model = ChatOpenAI(model="gpt-4o", api_key=openai_api_key, temperature=0)

# Initialize memory buffer
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create structured chat agent
agent = create_structured_chat_agent(llm=model, tools=tools, prompt=prompt)

# Create agent executor
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent,
                                                    tools=tools,
                                                    verbose=True,
                                                    memory=memory,
                                                    handle_parsing_errors=True)

# Add initial system message to memory
initial_message = "You are an AI assistant that can provide helpful answers using available tools."
memory.chat_memory.add_message(SystemMessage(content=initial_message))

# Chat loop to interact with the user
while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break

    # Add user's message to the conversation memory
    memory.chat_memory.add_message(HumanMessage(content=user_input))

    # Invoke the agent with the user input and the current chat history
    response = agent_executor.invoke({"input": user_input})
    print("Bot:", response["output"])

    # Add the agent's response to the conversation memory
    memory.chat_memory.add_message(AIMessage(content=response["output"]))
