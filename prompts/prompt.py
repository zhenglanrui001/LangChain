from fastapi import FastAPI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langserve import add_routes

from init_llm import deepseek_llm

prompt = ChatPromptTemplate.from_template('请将{text}翻译为{language}')

parser = StrOutputParser()

chain = prompt | deepseek_llm | parser

print(chain.invoke({'text': '今天天气预报预测要下雨，出门记得带伞！', 'language': 'English'}))

# 部署服务器  lang-server
# 创建fastAPI的应用
app = FastAPI(title='我的LangChain服务', version='1.0.0', description='使用LangChain翻译任何语句的服务器')

# 添加路由
add_routes(
    app,
    chain,
    path='/translate',
)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='localhost', port=8000)