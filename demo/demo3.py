import os

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import chat_agent_executor

from env_utils import LANGCHAIN_API_KEY, TAVILY_API_KEY
from init_llm import deepseek_llm

# 无法使用OpenAIEmbeddings()  -->  使用国内千文   -->  关闭代理服务器
os.environ['http_proxy'] = 'http://127.0.0.1:33210'
os.environ['https_proxy'] = 'http://127.0.0.1:33210'

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = 'ChatRobot'
os.environ['LANGCHAIN_API_KEY'] = LANGCHAIN_API_KEY
os.environ['TAVILY_API_KEY'] = TAVILY_API_KEY


os.environ.pop('https_proxy', None)
os.environ.pop('http_proxy', None)
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)


# 代理构建

# LLM
model = deepseek_llm

# 没有代理
# result = model.invoke([HumanMessage(content='北京天气怎么样？')])
# print(result)

# LangChain内置工具，使用Tavily搜索引擎作为工具
search = TavilySearchResults(max_results=3)  # 搜索结果: 3
# print(search.invoke('北京天气怎么样？'))

# 模型绑定工具
tools = [search]
# model_with_tools = model.bind_tools(tools)

# 模型自动推理是否需要调用工具完成用户问题 --> 没有真正执行工具
# resp1 = model_with_tools.invoke('中国未来3年计算机专业学生的就业率')
#
# print(f'Model_Result_Content: {resp1.content}')
# print(f'Model_Result_Content: {resp1.tool_calls}')
#
# print()
#
# resp2 = model_with_tools.invoke('中国未来3年计算机专业本科与研究生学生平均薪资')
#
# print(f'Model_Result_Content: {resp2.content}')
# print(f'Model_Result_Content: {resp2.tool_calls}')

# 创建代理
agent_executor = chat_agent_executor.create_tool_calling_executor(model, tools)

resp1 = agent_executor.invoke({'messages': [HumanMessage(content='中国未来3年计算机专业学生的就业率')]})
print(resp1['messages'][-1].content)

print()

resp2 = agent_executor.invoke({'messages': [HumanMessage(content='中国未来3年计算机专业本科与研究生学生平均薪资')]})
print(resp2['messages'][-1].content)

print()

resp3 = agent_executor.invoke({'messages': [HumanMessage(content='中国的首都是哪个城市？')]})
print(resp3['messages'][-1].content)