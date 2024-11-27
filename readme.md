# 使用 GraphRAG 和 OpenCompass 进行评测， 以及对于GraphRAG进行优化

## 环境配置

除了 GraphRAG 和 OpenCompass 的 conda 环境配置外，为了使用 API 测评，还需要额外安装以下依赖：

`pip install "opencompass[api]"`

### 配置参考

成功配置用于 GraphRAG 的环境，可以参考以下两个视频进行搭建：
- [GraphRAG 配置视频 1](https://www.bilibili.com/video/BV1HmWQeKEQB/?spm_id_from=..search-card.all.click&vd_source=dd722f9cbe299d2a0d932b162f651357)
- [GraphRAG 配置视频 2](https://www.bilibili.com/video/BV1aKWTexEeH/?vd_source=30acb5331e4f5739ebbad50f7cc6b949)

成功配置 OpenCompass 环境，除了 OpenCompass 自身必要的安装以外，还需要额外安装对 `data` 和 `api` 的支持。这些内容可以参考 OpenCompass 的[说明文档](https://github.com/open-compass/opencompass)。

---

## 启动和配置

要正常运行 GraphRAG 的测评，首先需要启动`one-api`，并在 GraphRAG 文件夹中配置 API。

### GraphRAG 文件夹的配置

- **`.env` 文件**  
  需要将`GRAPHRAG_CHAT_API_KEY`以及`GRAPHRAG_EMBEDDING_API_KEY`替换为实际的 API key。

- **`settings.yaml` 文件**  
  该文件中的参数已经针对 Zhipu 模型进行了调整。如果使用其他模型，需要进行相应的调整。

- **`utils/main.py` 文件**  
  - 修改`setup_llm_and_embedder`函数中的 API key 为实际的 API key。  
  - 将路径调整到本地路径`GraphRAGTest/ragtest/inputs/artifacts`（第 48 行）。  
  - 确保第 110 行和第 119 行的 API key 已替换为 OneAPI 转化后的 API key。

### OpenCompass 文件夹的配置

- 通过运行`run.sh`启动测评，其中`eval_api_zhipu_v2.py`是对 Zhipu 模型的普通评测。  
  如果需要使用该评测，需要修改`opencompass/configs/api_examples/eval_api_zhipu_v2.py`中的 API key。

- 运行`configs/eval_myTest.py`会对 GraphRAG 进行评测，但需要提前启动`one-api`服务。

---

## GraphRAG 测评文件说明

为了执行 GraphRAG 的评测，主要的文件是`opencompass/opencompass/models/GraphRAGModel.py`，其包含以下功能：

- **拆分 prompt 为知识部分和问题部分的代码**  
  将输入的 prompt 拆分为知识内容和问题部分，以便更好地组织检索。

- **将拆出的知识部分存入 GraphRAG 中的代码**  
  确保知识被有效存储以供检索。

- **运行索引构建的代码**  
  当前设定最大 7 次重试以确保索引构建成功， 这是因为索引构建过程可能失败。
  **注意**：需要在`_run_indexing_command()`中重新配置索引构建命令的路径。

- **启动 GraphRAG 接收问题 API 的代码**  
  该部分代码会额外开启一个命令窗口等待问题输入。  
  **注意**：需在`_start_api_service()`中重新配置`api_command`。

- **发送问题并接收答案的代码**  
  将问题通过 API 发送到服务端并接收模型返回的结果。

- **关闭 GraphRAG 接收问题 API 的代码**  
  确保在测试结束后关闭服务。

## 当前问题与改进方向

目前存在的问题：  
- 使用 GraphRAG 产出的评分不如直接使用 Zhipu 模型。  不过则还是因为当前我使用的TriviaQA数据集不是特别适合RAG相关的测试， 因为内容中并不能100%保证一定存在正确答案， 而RAG的特性就是仅根据知识文档中的信息回答， 所以对于一些问题无法回答导致评分低于普通zhipu模型。
- 对于部分复杂问题，GraphRAG 构建的知识图谱存在指代消解不准确的问题。

改进方向：  
1. **指代消解问题**  
   增强对指代关系的识别能力，解决上一句为人名、下一句为代词时无法正确指向的问题。

2. **段落级图谱构建**  
   当前的图谱构建基于句子，后续可探索按段落构建更整体且具有关联性的图谱，提升对上下文的理解能力。

