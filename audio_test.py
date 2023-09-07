import openai
import os
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai.organization = "org-Q2q2eUsNbWG6pxm1dbt2N5uW"
openai.api_key = OPENAI_API_KEY



# audio_file= open("20230701_174422.m4a", "rb")
# transcript = openai.Audio.translate("whisper-1", audio_file)
# print(transcript["text"])

from langchain.llms import OpenAI
#gpt-3.5-turbo
#text-davinci-003
llm = OpenAI(openai_api_key=OPENAI_API_KEY,model_name="gpt-3.5-turbo")


# data = openai.Completion.create(
#     model="text-davinci-003",
#     prompt="Tell me a Joke"
# )

# print(llm("Tell me a Joke"))

# result = llm("生成可执行的快速排序 Python 代码")
# print(result)

import openai

data = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        {"role": "user", "content": "Where was it played?"}
    ]
)

print(data)