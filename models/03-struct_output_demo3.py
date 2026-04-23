'''
jsonschema 结构化输出

'''
from typing import TypedDict, Annotated

from init_llm import deepseek_llm


json_schema = {
    'title': 'movie',
    'description': '电影的详细信息，包括标题，上映日期，导演与评分',
    'type': 'object',
    'properties': {
        'title': {'type': 'string'},
        'year': {'type': 'integer'},
        'director': {'type': 'string'},
        'rating': {'type': 'number'}
    }
}


model_with_structured_output = deepseek_llm.with_structured_output(json_schema)

resp = model_with_structured_output.invoke('请介绍电影《坠落的审判》')
print(type(resp))
print(resp)
