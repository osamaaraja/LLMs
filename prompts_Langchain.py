from langchain.prompts import ChatPromptTemplate

# creating a prompt template with one value
template = "Tell me a joke about {topic}."
prompt_template = ChatPromptTemplate.from_template(template=template)

print("------- Prompt from Template -------")
prompt = prompt_template.invoke({"topic": "cat"})
print(prompt)
print("-"*100)


# creating a prompt template with multiple values
template_multiple = ("""You are a helpful assistant.
Human: Tell me a {adjective} story about a {animal}.
Assistant:""")

prompt_multiple = ChatPromptTemplate.from_template(template=template_multiple)
prompt = prompt_multiple.invoke({"adjective": "funny", "animal": "panda"})
print("\n------ Prompt with Multple Placeholders ------\n")
print(prompt)
print("-"*100)

# prompting with system and human messages
message = [("system", "You are a comedian who tells joke about {topic}."),("human","Tell me {joke_count} jokes.")]
prompt_template = ChatPromptTemplate.from_messages(messages=message)
prompt = prompt_template.invoke({"topic": "cat", "joke_count": "3"})
print("\n------ Prompt with system and human messages ------\n")
print(prompt)
print("-"*100)
