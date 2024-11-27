from mmengine.config import read_base


from opencompass.models.GraphRAGModel import GraphRAGModel

# /home/kuro/Desktop/NTU13Oct/modify/opencompass/opencompass/models/GraphRAGModel.py


with read_base():
    from opencompass.configs.datasets.triviaqarc.triviaqarc_gen_db6413 import (
        triviaqarc_datasets,
    )

datasets = triviaqarc_datasets

models = [
    dict(
        abbr="graphrag_model",
        type=GraphRAGModel,
        api_url="http://localhost:8012/v1/chat/completions",  # GraphRAG API 地址
        input_dir="/home/kuro/Desktop/NTU13Oct/modify/GraphragTest/ragtest/input",  # 这个需要配置为本地的graphrag的input文件夹
    )
]

work_dir = "./outputs/myNewTest/"
