# configs/custom_imports.py

custom_imports = dict(
    imports=["opencompass.models.GraphRAGModel"], allow_failed_imports=False
)

models = [
    dict(
        abbr="graphrag_model",
        type="GraphRAGModel",
        api_url="http://localhost:8012/v1/chat/completions",
        input_dir="/home/kuro/Desktop/GT/GraphTest/ragtest/input",
    )
]
