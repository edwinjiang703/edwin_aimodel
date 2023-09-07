
from langchain.utilities import SerpAPIWrapper
from langchain.agents import Tool
from langchain.tools.file_management.write import WriteFileTool
from langchain.tools.file_management.read import ReadFileTool
from langchain.embeddings import OpenAIEmbeddings

import os

#os.environ["SERPAPI_API_KEY"] = "e65622355785aba531fe0f3733c6c429e3ec43457c916a0c3006e6f81d433369"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print(OPENAI_API_KEY)
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
print(SERPAPI_API_KEY)


# 构造 AutoGPT 的工具集
search = SerpAPIWrapper()
tools = [
    Tool(
        name="search",
        func=search.run,
        description="useful for when you need to answer questions about current events. You should ask targeted questions",
    ),
    WriteFileTool(),
    ReadFileTool(),
]


embeddings_model = OpenAIEmbeddings()


import faiss
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore

# OpenAI Embedding 向量维数
embedding_size = 1536
# 使用 Faiss 的 IndexFlatL2 索引
index = faiss.IndexFlatL2(embedding_size)
# 实例化 Faiss 向量数据库
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})



from langchain_experimental.autonomous_agents import AutoGPT
from langchain.chat_models import ChatOpenAI

agent = AutoGPT.from_llm_and_tools(
    ai_name="Jarvis",
    ai_role="Assistant",
    tools=tools,
    llm=ChatOpenAI(temperature=0),
    memory=vectorstore.as_retriever(), # 实例化 Faiss 的 VectorStoreRetriever
)

# agent.chain.verbose = True
agent.run(["2023年大运会举办地在哪？", "2023年大运会中国队奖牌数多少？"])