from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
import os

load_dotenv() # loading the environment variables

openai_api_key= os.getenv('OPEN_API_KEY') # getting the API key from the .env file

model = ChatOpenAI(model='gpt-4o', api_key=openai_api_key)


print("------- Prompt from Template -------")
# creating a prompt template with one value
template = "Tell me a joke about {topic}."
prompt_template = ChatPromptTemplate.from_template(template=template)
prompt = prompt_template.invoke({"topic": "cats"})
print(f"Prompt: {prompt}")
result = model.invoke(prompt) # invoking the model to get the output from the model
print(f"Result: {result.content}")
print("-"*100)

# creating a prompt template with multiple values
print("------ Prompt with Multple Placeholders ------")
template_multiple = ("""You are a helpful assistant.
Human: Tell me a {adjective} story about a {animal}.
Assistant:""")
prompt_multiple = ChatPromptTemplate.from_template(template=template_multiple)
prompt = prompt_multiple.invoke({"adjective": "funny", "animal": "panda"})
print(f"Prompt: {prompt}")
result = model.invoke(prompt) # invoking the model to get the output from the model
print(f"Result: {result.content}")
print("-"*100)


# prompting with system and human messages
print("------ Prompt with system and human messages ------")
message = [("system", "You are a comedian who tells joke about {topic}."),("human","Tell me {joke_count} jokes.")]
prompt_template = ChatPromptTemplate.from_messages(messages=message)
prompt = prompt_template.invoke({"topic": "cat", "joke_count": "3"})
print(f"Prompt: {prompt}")
result = model.invoke(prompt) # invoking the model to get the output from the model
print(f"Result: {result.content}")
print("-"*100)