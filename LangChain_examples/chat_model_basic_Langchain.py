
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os

load_dotenv() # loading the environment variables
openai_api_key = os.getenv('OPEN_API_KEY')

# Creating the model
model = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key)

# invoking the model
result = model.invoke("There are 3 red, 3 blue and 4 green balls. what is the probability of picking a red ball?")
print("Full result: ", result, "\nContent-Only:", result.content)