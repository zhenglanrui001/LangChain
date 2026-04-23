'''
pydantic 模型结构化输出

'''
from pydantic import BaseModel, Field

from init_llm import deepseek_llm


class Movie(BaseModel):
    title: str = Field(description='电影标题')
    year: str = Field(description='电影上映年份')
    director: str = Field(description='电影导演, 中文名字， 例如：弗兰克·德拉邦特')
    rating: float = Field(description='电影评分')


model_with_structured_output = deepseek_llm.with_structured_output(Movie)

resp = model_with_structured_output.invoke('请介绍电影《坠落的审判》')
print(type(resp))
print(resp)
