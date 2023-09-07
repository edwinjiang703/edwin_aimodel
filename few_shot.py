from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import FewShotPromptTemplate, PromptTemplate,FewShotChatMessagePromptTemplate,ChatPromptTemplate
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 定义一个提示模板
example_prompt = PromptTemplate(
    input_variables=["input", "output"],     # 输入变量的名字
    template="Input: {input}\nOutput: {output}",  # 实际的模板字符串
)

# 这是一个假设的任务示例列表，用于创建反义词
examples = [
    {"input": "happy","output": "sad"},
    {"input": "tall","output": "short"},
    {"input": "energetic","output": "lethargic"},
    {"input": "sunny","output": "gloomy"},
    {"input": "windy","output": "calm"},
]

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,                          # 可供选择的示例列表
    OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY),                # 用于生成嵌入向量的嵌入类，用于衡量语义相似性
    Chroma,                            # 用于存储嵌入向量并进行相似性搜索的 VectorStore 类
    k=1                            # 要生成的示例数量
)

# 创建一个 FewShotPromptTemplate 对象
# similar_prompt = FewShotPromptTemplate(
#     example_selector=example_selector,  # 提供一个 ExampleSelector 替代示例
#     example_prompt=example_prompt,      # 前面定义的提示模板
#     prefix="Give the antonym of every input", # 前缀模板
#     suffix="Input: {adjective}\nOutput:",     # 后缀模板
#     input_variables=["adjective"],           # 输入变量的名字
# )
# print(similar_prompt.format(adjective="happy"))


few_shot_chat_prompt = FewShotChatMessagePromptTemplate(
    input_variables=["input"],
    example_selector=example_selector,
    example_prompt=ChatPromptTemplate.from_messages(
        [("human","{input}"),("ai","{output}")]
    ),
)

from langchain.chat_models import ChatAnthropic
from langchain.prompts import ChatPromptTemplate

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are wonderous wizard of language."),
        few_shot_chat_prompt,
        ("human", "{input}"),
    ]
)

chain = final_prompt
print(chain.invoke({"input": "long"}))

# from langchain.chat_models import ChatOpenAI
# chat_model = ChatOpenAI(model_name="gpt-3.5-turbo", max_tokens=1000,openai_api_key=OPENAI_API_KEY)

# messages=final_prompt.format_messages(input="windy")

# chat_result = chat_model(messages) 
# print(chat_result.content)
