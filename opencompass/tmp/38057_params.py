datasets = [
    [
        dict(
            abbr='triviaqarc',
            eval_cfg=dict(
                evaluator=dict(type='opencompass.datasets.TriviaQAEvaluator'),
                pred_role='BOT'),
            infer_cfg=dict(
                inferencer=dict(
                    batch_size=4,
                    max_out_len=50,
                    max_seq_len=8192,
                    type='opencompass.openicl.icl_inferencer.GenInferencer'),
                prompt_template=dict(
                    template=dict(round=[
                        dict(
                            prompt=
                            '{evidence}\nAnswer these questions:\nQ: {question}?A:',
                            role='HUMAN'),
                        dict(prompt='', role='BOT'),
                    ]),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    type='opencompass.openicl.icl_retriever.ZeroRetriever')),
            path='./data/triviaqa-rc/',
            reader_cfg=dict(
                input_columns=[
                    'question',
                    'evidence',
                ],
                output_column='answer',
                test_split='dev',
                train_split='dev'),
            type='opencompass.datasets.TriviaQArcDataset'),
    ],
]
models = [
    dict(
        abbr='graphrag_model',
        api_url='http://localhost:8012/v1/chat/completions',
        input_dir=
        '/home/kuro/Desktop/NTU13Oct/modify/GraphragTest/ragtest/input',
        type='opencompass.models.GraphRAGModel.GraphRAGModel'),
]
work_dir = './outputs/myNewTest/20241105_154754'
