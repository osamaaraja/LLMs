from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.schema import SystemMessage, HumanMessage, AIMessage
import os

load_dotenv() # loading the environment variables

openai_api_key= os.getenv('OPEN_API_KEY') # getting the API key from the .env file

model = ChatOpenAI(model='gpt-4o', api_key=openai_api_key)

chat_hist = [] # for storing the messages

system_message = SystemMessage(content="you are a helpful AI assistant")
chat_hist.append(system_message) # appending the messges into the chat history

while True:
    query = input("You: ")
    if query.lower() == 'exit': # if the user gives the keyword exit then close the chat
        break
    chat_hist.append(HumanMessage(content=query)) # adding user message

    result = model.invoke(chat_hist)
    response = result.content
    chat_hist.append(AIMessage(content=response))

    print(f"AI response: {response}")

print("---------------- Message History ------------------")

# print the entire chat history upon exit
for item in chat_hist:
    content = item.content
    parts = content.split(',')

    for part in parts:
        print(part.strip())






