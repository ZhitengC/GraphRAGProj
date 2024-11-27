from mmengine.config import read_base
from opencompass.models import ZhiPuV2AI


with read_base():
    #     from opencompass.configs.datasets.triviaqarc.triviaqarc_gen_db6413 import (
    #         triviaqarc_datasets,
    #     )

    # datasets = triviaqarc_datasets

    # /home/kuro/Desktop/NTU13Oct/modify/opencompass/configs/datasets/longbench/longbench2wikimqa/longbench_2wikimqa_gen_6b3efc.py
    from opencompass.configs.datasets.longbench.longbench2wikimqa.longbench_2wikimqa_gen_6b3efc import (
        LongBench_2wikimqa_datasets,
    )

datasets = LongBench_2wikimqa_datasets


api_meta_template = dict(
    round=[
        dict(role="HUMAN", api_role="HUMAN"),
        dict(role="BOT", api_role="BOT", generate=True),
    ],
)

models = [
    dict(
        abbr="glm4_notools",
        type=ZhiPuV2AI,
        path="glm-4",
        key="c4e94b960c2ba9e12e20b609db8d0417.r7Yp7ahcJ7o4DlJ7",
        generation_kwargs={
            "tools": [
                {
                    "type": "web_search",
                    "web_search": {
                        "enable": False  # turn off the search
                    },
                }
            ]
        },
        meta_template=api_meta_template,
        query_per_second=1,
        max_out_len=2048,
        max_seq_len=2048,
        batch_size=8,
    )
]


work_dir = "outputs/api_zhipu/"
