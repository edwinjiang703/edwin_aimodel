from langchain import PromptTemplate
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

prompt_template = PromptTemplate.from_template(
    "Tell me a {adjective} joke about {content}."
)

# 使用 format 生成提示
prompt = prompt_template.format(adjective="funny", content="chickens")
print(prompt)

prompt_template = PromptTemplate.from_template(
    "讲{num}个给程序员听得笑话"
)
from langchain.llms import OpenAI

llm = OpenAI(openai_api_key=OPENAI_API_KEY,model_name="text-davinci-003", max_tokens=1000)

prompt = prompt_template.format(num=2)
print(f"prompt: {prompt}")

result = llm(prompt)
print(f"result: {result}")


