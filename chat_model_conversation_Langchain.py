from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# system message - broad context for the conversation
# human message - us talking to the AI
# AI message - response from the AI
from langchain.schema import SystemMessage, HumanMessage, AIMessage


load_dotenv() # loading the environment variables

openai_api_key= os.getenv('OPEN_API_KEY') # getting the API key from the .env file

model = ChatOpenAI(model='gpt-4o', api_key=openai_api_key)

# creating a conversation
message = [
    SystemMessage(content="Solve the following probability problems"),
    HumanMessage(content="There are 3 red, 3 blue and 4 green balls. what is the probability of picking a red ball?"),
]

# invoking the model
result = model.invoke(message)
print(f"Answer from AI: {result.content}")
print("-"*100)


# the conversation can be continued with the AI
# creating a conversation to ge the context and mybe improving further on it
message = [
    SystemMessage(content="Solve the following probability problems"),
    HumanMessage(content="There are 3 red, 3 blue and 4 green balls. what is the probability of picking a red ball?"),
    AIMessage(content="the probability of picking a red ball is 0.3 or 30%."),
    HumanMessage(content="What is probability of picking a green ball?")
]

# invoking the model
result = model.invoke(message)
print(f"Answer from AI: {result.content}")