from langchain.chains.router import MultiPromptChain
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

physics_template = """你是一位非常聪明的物理教授。
你擅长以简洁易懂的方式回答关于物理的问题。
当你不知道某个问题的答案时，你会坦诚承认。

这是一个问题：
{input}"""

math_template = """你是一位很棒的数学家。你擅长回答数学问题。
之所以如此出色，是因为你能够将难题分解成各个组成部分，
先回答这些组成部分，然后再将它们整合起来回答更广泛的问题。

这是一个问题：
{input}"""

chemical_template =  """你是一位非常聪明的化学教授。
你擅长以简洁易懂的方式回答关于化学的问题。
当你不知道某个问题的答案时，你会坦诚承认。

这是一个问题：
{input}"""

db_sql =  """你是一位非常聪明的关于数据库方面的SQL语言专家。
你擅长以简洁易懂的方式回答关于数据库SQL方面的问题。
当你不知道某个问题的答案时，你会坦诚承认。
酶促应该以如下模式进行回复，例如：

问题描述：如下sql写法是否正确？
         select name from customers and orders
问题解答：不符合SQL的语法，应该写成 select name from customers ,orders where 两个表的关联条件


这是一个问题：
{input}"""


db_os =  """你是一位非常聪明的关于操作系统方面的专家。
你擅长以简洁易懂的方式回答关于操作系统方面的问题。
当你不知道某个问题的答案时，你会坦诚承认。

这是一个问题：
{input}"""

prompt_infos = [
    {
        "name":"物理",
        "description":"适用于回答物理问题",
        "prompt_template":physics_template,
    },
    {
        "name": "数学",
        "description": "适用于回答数学问题",
        "prompt_template": math_template,
    },
    {
        "name": "化学",
        "description": "适用于回答化学问题",
        "prompt_template": chemical_template,
    },
    {
        "name": "SQL",
        "description": "适用于回答关于数据库方面的SQL问题",
        "prompt_template": db_sql,
    },
    {
        "name": "OS",
        "description": "适用于回答关于操作系统问题",
        "prompt_template": db_os,
    },
]

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY,model_name="gpt-3.5-turbo")

# 创建一个空的目标链字典，用于存放根据prompt_infos生成的LLMChain。
destination_chains = {}

# 遍历prompt_infos列表，为每个信息创建一个LLMChain。
for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt= PromptTemplate(template=prompt_template,input_variables=["input"])
    chain = LLMChain(llm=llm,prompt=prompt)
    destination_chains[name] = chain

default_chain = ConversationChain(llm=llm,output_key="text")

from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE

destionations = [f"{p['name']}:{p['description']}" for p in prompt_infos]

destinations_str = "\n".join(destionations)

router_templates = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)

router_prompt = PromptTemplate(
    template=router_templates,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)

router_chain = LLMRouterChain.from_llm(llm,router_prompt)

print(destinations_str)

chain = MultiPromptChain(
    router_chain = router_chain,
    destination_chains = destination_chains,
    default_chain = default_chain,
    verbose = True,
)

print(chain.run("select * from dual  这个语句什么意思？"))