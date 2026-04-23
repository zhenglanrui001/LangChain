from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from init_llm import deepseek_llm


# LLM使用工具返回结果的步骤
# 1. 定义工具，并给LLM绑定工具
# 2. 与LLM进行对话，LLM返回大模型请求，并不会主动调用工具
# 3. 根据返回结果手动处理，并将结果告知LLM
# 4. LLM最后生成回复


@tool
def get_stock_price(company: str, timeframe: str = "today") -> str:
    """获取指定公司的股票价格信息
       模拟数据库
    Args:
        company: 公司名称（如：苹果公司, 微软公司, 谷歌公司）
        timeframe: 时间范围（today-今日, week-本周, month-本月）
    """
    # 模拟股票数据
    mock_data = {
        "苹果公司": {"today": 185.20, "week": 183.50, "month": 180.75},
        "微软公司": {"today": 415.86, "week": 412.30, "month": 405.42},
        "谷歌公司": {"today": 15.42, "week": 15.20, "month": 14.85}
    }

    if company in mock_data:
        price = mock_data[company].get(timeframe, "未知时间范围")
        return f"{company} {timeframe}价格: {price}美元"
    else:
        return f"未找到股票代码 {company} 的数据"


# 定义新闻搜索工具
@tool
def search_news(company: str) -> str:
    """搜索指定公司的财经新闻
    Args:
        company: 公司名称
    Return:
        公司的财经新闻，每个新闻占一行
    """
    # 模拟新闻数据
    mock_news = {
        "苹果公司": [
            "苹果发布新款iPhone，股价上涨3%",
            "苹果与欧盟达成反垄断和解协议",
            "苹果将在印度扩大生产规模"
        ],
        "微软公司": [
            "微软Azure云业务季度增长超预期",
            "微软完成对Nuance的收购",
            "微软推出新一代AI助手Copilot"
        ],
        "谷歌公司": [
            "谷歌发布新AI模型，性能提升20%",
            "谷歌与OpenAI合作，开发新的AI助手",
            "谷歌在欧洲展开AI研究项目"
        ]
    }

    news_list = mock_news.get(company, [f"未找到{company}的相关新闻"])
    return "\n".join(news_list)


# 1. 绑定工具
model_bind_tool = deepseek_llm.bind_tools([get_stock_price, search_news])

messages = []
human_message = HumanMessage(content="比较苹果公司与谷歌公司本周股价")
messages.append(human_message)

# 2. LLM返回调用工具请求
while True:
    response = model_bind_tool.invoke(messages)
    messages.append(response)

    if response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call['name'] == 'get_stock_price':
                # 3. 根据返回结果手动处理，并将结果告知LLM
                tool_result = get_stock_price.invoke(tool_call)
                messages.append(tool_result)
            elif tool_call['name'] == 'search_news':
                tool_result = search_news.invoke(tool_call)
                messages.append(tool_result)
    else:
        print('没有工具调用')
        break


print('messages', messages)
print('response', response)