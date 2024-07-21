from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv() # loading the environment variables

openai_api_key= os.getenv('OPEN_API_KEY') # getting the API key from the .env file

model = ChatOpenAI(model='gpt-4o', api_key=openai_api_key)


prompt_template = ChatPromptTemplate.from_messages([("system", "You are a comedian who tells jokes about {topic}."),
                                                    ("human", "Tell me {joke_count} jokes.")]) # messages are used as tuples

# defining the chain
chain = prompt_template | model | StrOutputParser()

# invoking the chain
result = chain.invoke({"topic": "humans", "joke_count": 3})
print(result)
