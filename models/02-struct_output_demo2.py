'''
typedict 字典模型结构化输出

'''
from typing import TypedDict, Annotated

from init_llm import deepseek_llm


class Movie(TypedDict):
    title: Annotated[str, '电影标题']
    year: Annotated[int, '电影上映年份']
    director: Annotated[str, '电影导演, 中文名字，例如：弗兰克·德拉邦特']
    rating: Annotated[float, '电影评分']


model_with_structured_output = deepseek_llm.with_structured_output(Movie)

resp = model_with_structured_output.invoke('请介绍电影《坠落的审判》')
print(type(resp))
print(resp)
