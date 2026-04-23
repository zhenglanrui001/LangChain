import os

from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from init_llm import deepseek_llm

# 无法使用OpenAIEmbeddings()  -->  使用国内千文   -->  关闭代理服务器
# os.environ['http_proxy'] = 'http://127.0.0.1:33210'
# os.environ['https_proxy'] = 'http://127.0.0.1:33210'

# os.environ['LANGCHAIN_TRACING_V2'] = 'true'
# os.environ['LANGCHAIN_PROJECT'] = 'ChatRobot'


os.environ.pop('https_proxy', None)
os.environ.pop('http_proxy', None)
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)

# LLM
model = deepseek_llm

# 准备测试数据
documents = [
    Document(
        page_content='小狗是伟大的伴侣，以其忠诚和友好而闻名。',
        metadata={'source': '哺乳动物宠物文档'}
    ),
    Document(
        page_content='猫是独立的动物，通常喜欢自己的空间。',
        metadata={'source': '哺乳动物宠物文档'}
    ),
    Document(
        page_content='金鱼是初学者的流行动物，护理相对简单。',
        metadata={'source': '哺乳动物宠物文档'}
    ),
    Document(
        page_content='鹦鹉是聪明的鸟类，能够模仿人类语言。',
        metadata={'source': '哺乳动物宠物文档'}
    ),
    Document(
        page_content='兔子是社交动物，需要足够的空间跳跃。',
        metadata={'source': '哺乳动物宠物文档'}
    )
]

# 实例化一个向量数据库  非 Runnable对象
vector_db = Chroma.from_documents(documents, embedding=DashScopeEmbeddings(),
                                  collection_metadata={'hnsw:space': 'cosine'})

# 相似度查询  分越低，相似度越高 --> 实际上返回的是夹角距离，即：cosine_distance = 1 - cosine_similarity
# print(vector_db.similarity_search_with_score('咖啡猫'))

# output:
# [(Document(id='0d492fce-3d94-43ce-acbd-2a7cf6a1b679', metadata={'source': '哺乳动物宠物文档'}, page_content='猫是独立的动物，通常喜欢自己的空间。'), 0.5877972841262817),
# (Document(id='0baef99e-232e-4580-8ab4-ebd930d33c34', metadata={'source': '哺乳动物宠物文档'}, page_content='小狗是伟大的伴侣，以其忠诚和友好而闻名。'), 0.7808572053909302),
# (Document(id='f1c3dd74-9894-4b3a-b4a5-51cecc277efd', metadata={'source': '哺乳动物宠物文档'}, page_content='鹦鹉是聪明的鸟类，能够模仿人类语言。'), 0.8155195713043213),
# (Document(id='efb39113-53e7-4c88-8745-aa7e7d872abb', metadata={'source': '哺乳动物宠物文档'}, page_content='兔子是社交动物，需要足够的空间跳跃。'), 0.829312801361084)]


# 检索器: bind(k=1) 返回相似度最高的第一个
retriever = RunnableLambda(vector_db.similarity_search).bind(k=1)

# print(retriever.batch(['咖啡猫', '鲨鱼']))
# output:
# [[Document(id='b7047da6-fda7-436a-a144-ad1eb30f8d41', metadata={'source': '哺乳动物宠物文档'}, page_content='猫是独立的动物，通常喜欢自己的空间。')],
# [Document(id='195ea58d-c533-4e30-9029-b9f337e72e46', metadata={'source': '哺乳动物宠物文档'}, page_content='金鱼是初学者的流行动物，护理相对简单。')]]

# prompt
message = """
使用提供的上下文仅回答这个问题:
{question}
上下文:
{context}
"""

prompt = ChatPromptTemplate.from_messages([('human', message)])

# chain
# RunnablePassthrough 允许之后再将用户问题传递给prompt和llm
chain = {'question': RunnablePassthrough(), 'context': retriever} | prompt | deepseek_llm

resp = chain.invoke('请介绍以下小猫。')
print(resp.content)


# 升级为真实可用RAG系统
# 1.数据document不再写死
# 2.向量DB持久化  -->  vector_db = Chroma(persist_dict='./db', embedding_function=DashScopeEmbeddings())
# 3.retriever升级 --> retriever = vector_db.as_retriever(search_type="mmr",
#     search_kwargs={"k": 3, "fetch_k": 10})
# 4. prompt升级 --> prompt = ChatPromptTemplate.from_messages([
#     ("system",
#      "你是一个专业AI助手。"
#      "只能使用给定上下文回答问题。"
#      "如果上下文没有答案，请回答“我不知道”。"),
#
#     ("human",
#      "问题：{question}\n\n"
#      "上下文：{context}")
# ])

# 5.RAG标准chain
# from langchain_core.runnables import RunnableParallel
# chain = (
#     RunnableParallel({
#         "context": retriever,
#         "question": RunnablePassthrough()
#     })
#     | prompt
#     | deepseek_llm
# )

# 6.部署
# from fastapi import FastAPI
# from langserve import add_routes
#
# app = FastAPI()
#
# add_routes(app, chain, path="/rag")
#
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)