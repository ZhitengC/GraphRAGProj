# 使用 GraphRAG 和 OpenCompass 进行 API 评测

## 环境配置

### 必要依赖

除了配置 GraphRAG 和 OpenCompass 的 Conda 环境外，还需额外安装以下依赖：

```bash
pip install "opencompass[api]"
配置参考
以下视频可帮助搭建环境：

GraphRAG 配置视频 1
GraphRAG 配置视频 2
启动和配置
GraphRAG 配置
.env 文件
配置以下环境变量：

plaintext
Copy code
GRAPHRAG_CHAT_API_KEY=<your_chat_api_key>
GRAPHRAG_EMBEDDING_API_KEY=<your_embedding_api_key>
settings.yaml 文件
参数已针对 Zhipu 模型调整。如果使用其他模型，请根据需求修改。

utils/main.py 文件

路径调整：将第 48 行中的路径替换为本地路径 GraphRAGTest/ragtest/inputs/artifacts。
API Key 修改：调整第 110 和 119 行的 API Key 为有效的 OneAPI 转化后的 API Key。
OpenCompass 配置
启动 通过以下命令启动 OpenCompass 测评：

bash
Copy code
bash run.sh
评测脚本

普通评测：运行 eval_api_zhipu_v2.py 测试 Zhipu 模型。
确保修改 opencompass/configs/api_examples/eval_api_zhipu_v2.py 中的 API Key。
GraphRAG 评测：运行 configs/eval_myTest.py 测试 GraphRAG。
注意：运行此脚本前需启动 OneAPI 服务。
GraphRAG 测评文件说明
主要评测代码位于 opencompass/opencompass/models/GraphRAGModel.py，功能包括：

Prompt 拆分
将 Prompt 拆分为知识部分和问题部分。

知识存储
将知识部分存入 GraphRAG 中。

索引构建

默认支持最大 7 次重试以确保索引构建成功。
修改索引路径：需在 _run_indexing_command() 中重新配置索引命令路径。
启动 API 服务

开启用于接收问题的 API 服务。
修改路径：需在 _start_api_service() 中重新配置 api_command。
问题传递和回答接收
发送问题并接收答案。

关闭 API 服务
确保 GraphRAG 的接收 API 服务在完成后关闭。

当前问题与改进方向
研究方向
理解 GraphRAG 原理
深入研究 GraphRAG 在知识存储与检索中的机制。

寻找更适合的数据集
使用高质量、多样性的数据集，优化模型表现。

改进建议
指代消解问题

解决如 “上一句提到人名，下一句包含 he/she/it” 的指代问题。
提升模型对代词指代的准确性。
段落级图谱构建

当前图谱构建可能基于单句。
探索按段落进行整体图谱构建，提高对上下文和关联性的理解。
配置验证步骤
配置 GraphRAG 和 OpenCompass 的 Conda 环境。
配置 .env 和 settings.yaml 文件中的 API Key。
确保 utils/main.py 文件中路径和 Key 已更新。
启动 OneAPI 服务。
运行 OpenCompass 测评脚本验证配置成功。
go
Copy code
```
