LongBench_2wikimqa_datasets=[
    dict(abbr='LongBench_2wikimqa',
        eval_cfg=dict(
            evaluator=dict(
                type='opencompass.datasets.LongBenchF1Evaluator'),
            pred_role='BOT'),
        infer_cfg=dict(
            inferencer=dict(
                max_out_len=32,
                type='opencompass.openicl.icl_inferencer.GenInferencer'),
            prompt_template=dict(
                template=dict(
                    round=[
                        dict(prompt='Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:',
                            role='HUMAN'),
                        ]),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                type='opencompass.openicl.icl_retriever.ZeroRetriever')),
        name='2wikimqa',
        path='opencompass/Longbench',
        reader_cfg=dict(
            input_columns=[
                'context',
                'input',
                ],
            output_column='answers',
            test_split='test',
            train_split='test'),
        type='opencompass.datasets.LongBench2wikimqaDataset'),
    ]
api_meta_template=dict(
    round=[
        dict(api_role='HUMAN',
            role='HUMAN'),
        dict(api_role='BOT',
            generate=True,
            role='BOT'),
        ])
datasets=[
    dict(abbr='LongBench_2wikimqa',
        eval_cfg=dict(
            evaluator=dict(
                type='opencompass.datasets.LongBenchF1Evaluator'),
            pred_role='BOT'),
        infer_cfg=dict(
            inferencer=dict(
                max_out_len=32,
                type='opencompass.openicl.icl_inferencer.GenInferencer'),
            prompt_template=dict(
                template=dict(
                    round=[
                        dict(prompt='Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:',
                            role='HUMAN'),
                        ]),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                type='opencompass.openicl.icl_retriever.ZeroRetriever')),
        name='2wikimqa',
        path='opencompass/Longbench',
        reader_cfg=dict(
            input_columns=[
                'context',
                'input',
                ],
            output_column='answers',
            test_split='test',
            train_split='test'),
        type='opencompass.datasets.LongBench2wikimqaDataset'),
    ]
models=[
    dict(abbr='glm4_notools',
        batch_size=8,
        generation_kwargs=dict(
            tools=[
                dict(type='web_search',
                    web_search=dict(
                        enable=False)),
                ]),
        key='c4e94b960c2ba9e12e20b609db8d0417.r7Yp7ahcJ7o4DlJ7',
        max_out_len=2048,
        max_seq_len=2048,
        meta_template=dict(
            round=[
                dict(api_role='HUMAN',
                    role='HUMAN'),
                dict(api_role='BOT',
                    generate=True,
                    role='BOT'),
                ]),
        path='glm-4',
        query_per_second=1,
        type='opencompass.models.ZhiPuV2AI'),
    ]
work_dir='outputs/api_zhipu/20241125_095332'