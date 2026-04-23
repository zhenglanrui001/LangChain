from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, wrap_model_call, ModelResponse, dynamic_prompt
from langchain.chat_models import init_chat_model
from langchain_core.messages import function, ToolMessage
from langchain_core.tools import tool

from env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL


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
    # 模拟错误，抛出异常
    # raise ValueError('股票接口不可用')


base_model = init_chat_model(
    model='deepseek-chat',
    model_provider='deepseek',
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL
)

advanced_model = init_chat_model(
    model='qwen-plus',
    model_provider='openai',
    api_key=DASHSCOPE_API_KEY,
    base_url=DASHSCOPE_BASE_URL
)


@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler: function) -> ModelResponse:
    ''' 根据用户输入动态选择模型 '''

    message_count = len(request.state['messages'])

    if message_count >= 3:
        model = advanced_model
    else:
        model = base_model

    return handler(request.override(model=model))


@dynamic_prompt
def dynamic_prompt(request: ModelRequest) -> str:
    ''' 根据用户类型使用不同提示词 '''
    user_type = request.runtime.context.get('user_type', 'normal')

    if user_type == 'vip':
        prompt = '回答问题之前，首先称呼：尊贵的vip客户您好，然后再回答用户问题。'
    else:
        prompt = '直接回答用户问题。'
    return prompt


# @wrap_model_call
# def handler_tool_errors(request, handler):
#     try:
#         return handler(request)
#     except Exception as e:
#         return ToolMessage(
#             tool_call_id=request.tool_call_id['id'],
#             content=f'目前工具调用不可用,错误信息：{str(e)}')


agent = create_agent(
    model=base_model,
    tools=[get_stock_price, search_news],
    middleware=[dynamic_model_selection, dynamic_prompt],
)

response = agent.invoke({'messages': [{'role': 'user', 'content': '查找谷歌公司股价新闻'}]},
                        context={'user_type': 'xxx'}
                        )

print(response)
print(response['messages'][-1].content)
