import regex as re
import tiktoken

gpt2pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+|[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""") # https://www.regular-expressions.info/quickstart.html

# splitting the string before encoding to avoid some merges across letters or punctuations or numbers
print(re.findall(gpt2pat, "Hello5656 world's!!!!       ")) # the string that we want to encode into tokens


# splitting a python code

code_example = """
for i in range(1,101):
    if i % 3 == 0 and i % 5 == 0:
        print("hello world")
    elif i % 3 == 0:
        print("hello")
    elif i % 5 == 0:
        print("world")
    else:
        print(i)
"""

print(re.findall(gpt2pat,code_example)) # the identation appears as spaces and don't get merged


# GPT-2 does not merge spaces
enc = tiktoken.get_encoding("gpt2")
print(f'GPT-2 result: {enc.encode("      hello world!!!")}')

# GPT-4 merges spaces
enc = tiktoken.get_encoding("cl100k_base")
print(f'GPT-4 result: {enc.encode("      hello world!!!")}') # gpt-4 is using a different pattern as written above