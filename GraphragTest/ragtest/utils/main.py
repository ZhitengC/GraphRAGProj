import os
import asyncio
import time
import uuid
import json
import re
import pandas as pd
import tiktoken
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from contextlib import asynccontextmanager
import uvicorn

# GraphRAG 相关导入
from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.indexer_adapters import (
    read_indexer_covariates,
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.query.input.loaders.dfs import store_entity_semantic_embeddings
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.embedding import OpenAIEmbedding
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.question_gen.local_gen import LocalQuestionGen
from graphrag.query.structured_search.local_search.mixed_context import (
    LocalSearchMixedContext,
)
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.query.structured_search.global_search.community_context import (
    GlobalCommunityContext,
)
from graphrag.query.structured_search.global_search.search import GlobalSearch
from graphrag.vector_stores.lancedb import LanceDBVectorStore


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


INPUT_DIR = ""
LANCEDB_URI = f"{INPUT_DIR}/lancedb"
COMMUNITY_REPORT_TABLE = "create_final_community_reports"
ENTITY_TABLE = "create_final_nodes"
ENTITY_EMBEDDING_TABLE = "create_final_entities"
RELATIONSHIP_TABLE = "create_final_relationships"
COVARIATE_TABLE = "create_final_covariates"
TEXT_UNIT_TABLE = "create_final_text_units"
COMMUNITY_LEVEL = 2
PORT = 8012

local_search_engine = None
global_search_engine = None
question_generator = None


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: Usage
    system_fingerprint: Optional[str] = None


async def setup_llm_and_embedder():
    logger.info("正在设置LLM和嵌入器")
    llm = ChatOpenAI(
        api_base="http://localhost:3000/v1",
        api_key="",  # 这个需要配制成实际的api_key
        model="glm-4",
        api_type=OpenaiApiType.OpenAI,
    )

    token_encoder = tiktoken.get_encoding("cl100k_base")

    text_embedder = OpenAIEmbedding(
        api_base="http://localhost:3000/v1",
        api_key="",  # 这个需要配制成实际的api_key
        model="embedding-2",
        deployment_name="embedding-2",
        api_type=OpenaiApiType.OpenAI,
        max_retries=20,
    )

    logger.info("LLM和嵌入器设置完成")
    return llm, token_encoder, text_embedder


async def load_context():
    logger.info("正在加载上下文数据")
    try:
        entity_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_TABLE}.parquet")
        entity_embedding_df = pd.read_parquet(
            f"{INPUT_DIR}/{ENTITY_EMBEDDING_TABLE}.parquet"
        )
        entities = read_indexer_entities(
            entity_df, entity_embedding_df, COMMUNITY_LEVEL
        )
        description_embedding_store = LanceDBVectorStore(
            collection_name="entity_description_embeddings"
        )
        description_embedding_store.connect(db_uri=LANCEDB_URI)
        store_entity_semantic_embeddings(
            entities=entities, vectorstore=description_embedding_store
        )
        relationship_df = pd.read_parquet(f"{INPUT_DIR}/{RELATIONSHIP_TABLE}.parquet")
        relationships = read_indexer_relationships(relationship_df)
        report_df = pd.read_parquet(f"{INPUT_DIR}/{COMMUNITY_REPORT_TABLE}.parquet")
        reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL)
        text_unit_df = pd.read_parquet(f"{INPUT_DIR}/{TEXT_UNIT_TABLE}.parquet")
        text_units = read_indexer_text_units(text_unit_df)
        covariate_df = pd.read_parquet(f"{INPUT_DIR}/{COVARIATE_TABLE}.parquet")
        claims = read_indexer_covariates(covariate_df)
        logger.info(f"声明记录数: {len(claims)}")
        covariates = {"claims": claims}
        logger.info("上下文数据加载完成")
        return (
            entities,
            relationships,
            reports,
            text_units,
            description_embedding_store,
            covariates,
        )
    except Exception as e:
        logger.error(f"加载上下文数据时出错: {str(e)}")
        raise


async def setup_search_engines(
    llm,
    token_encoder,
    text_embedder,
    entities,
    relationships,
    reports,
    text_units,
    description_embedding_store,
    covariates,
):
    logger.info("正在设置搜索引擎")
    local_context_builder = LocalSearchMixedContext(
        community_reports=reports,
        text_units=text_units,
        entities=entities,
        relationships=relationships,
        covariates=covariates,
        entity_text_embeddings=description_embedding_store,
        embedding_vectorstore_key=EntityVectorStoreKey.ID,
        text_embedder=text_embedder,
        token_encoder=token_encoder,
    )

    local_context_params = {
        "text_unit_prop": 0.5,
        "community_prop": 0.1,
        "conversation_history_max_turns": 5,
        "conversation_history_user_turns_only": True,
        "top_k_mapped_entities": 10,
        "top_k_relationships": 10,
        "include_entity_rank": True,
        "include_relationship_weight": True,
        "include_community_rank": False,
        "return_candidate_context": False,
        "embedding_vectorstore_key": EntityVectorStoreKey.ID,
        # "max_tokens": 12_000,
        "max_tokens": 4096,
    }

    local_llm_params = {
        # "max_tokens": 2_000,
        "max_tokens": 4096,
        "temperature": 0.0,
    }

    local_search_engine = LocalSearch(
        llm=llm,
        context_builder=local_context_builder,
        token_encoder=token_encoder,
        llm_params=local_llm_params,
        context_builder_params=local_context_params,
        response_type="multiple paragraphs",
    )

    global_context_builder = GlobalCommunityContext(
        community_reports=reports,
        entities=entities,
        token_encoder=token_encoder,
    )

    global_context_builder_params = {
        "use_community_summary": False,
        "shuffle_data": True,
        "include_community_rank": True,
        "min_community_rank": 0,
        "community_rank_name": "rank",
        "include_community_weight": True,
        "community_weight_name": "occurrence weight",
        "normalize_community_weight": True,
        # "max_tokens": 12_000,
        "max_tokens": 4096,
        "context_name": "Reports",
    }

    map_llm_params = {
        "max_tokens": 1000,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }

    reduce_llm_params = {
        "max_tokens": 2000,
        "temperature": 0.0,
    }

    global_search_engine = GlobalSearch(
        llm=llm,
        context_builder=global_context_builder,
        token_encoder=token_encoder,
        # max_data_tokens=12_000,
        max_data_tokens=4096,
        map_llm_params=map_llm_params,
        reduce_llm_params=reduce_llm_params,
        allow_general_knowledge=False,
        json_mode=True,
        context_builder_params=global_context_builder_params,
        concurrent_coroutines=32,
        response_type="multiple paragraphs",
    )

    logger.info("搜索引擎设置完成")
    return (
        local_search_engine,
        global_search_engine,
        local_context_builder,
        local_llm_params,
        local_context_params,
    )


def format_response(response):
    paragraphs = re.split(r"\n{2,}", response)
    formatted_paragraphs = []
    for para in paragraphs:
        if "```" in para:
            parts = para.split("```")
            for i, part in enumerate(parts):
                if i % 2 == 1:
                    parts[i] = f"\n```\n{part.strip()}\n```\n"
            para = "".join(parts)
        else:
            para = para.replace(". ", ".\n")
        formatted_paragraphs.append(para.strip())
    return "\n\n".join(formatted_paragraphs)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global local_search_engine, global_search_engine, question_generator
    try:
        logger.info("正在初始化搜索引擎和问题生成器...")
        llm, token_encoder, text_embedder = await setup_llm_and_embedder()
        (
            entities,
            relationships,
            reports,
            text_units,
            description_embedding_store,
            covariates,
        ) = await load_context()
        (
            local_search_engine,
            global_search_engine,
            local_context_builder,
            local_llm_params,
            local_context_params,
        ) = await setup_search_engines(
            llm,
            token_encoder,
            text_embedder,
            entities,
            relationships,
            reports,
            text_units,
            description_embedding_store,
            covariates,
        )
        question_generator = LocalQuestionGen(
            llm=llm,
            context_builder=local_context_builder,
            token_encoder=token_encoder,
            llm_params=local_llm_params,
            context_builder_params=local_context_params,
        )
        logger.info("初始化完成")
    except Exception as e:
        logger.error(f"初始化过程中出错: {str(e)}")
        raise
    yield

    logger.info("正在关闭...")


app = FastAPI(lifespan=lifespan)


async def full_model_search(prompt: str):
    local_result = await local_search_engine.asearch(prompt)
    global_result = await global_search_engine.asearch(prompt)
    formatted_result = "#综合搜索结果:\n\n"
    formatted_result += "##本地检索结果:\n"
    formatted_result += format_response(local_result.response) + "\n\n"
    formatted_result += "##全局检索结果:\n"
    formatted_result += format_response(global_result.response) + "\n\n"
    return formatted_result


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if not local_search_engine or not global_search_engine:
        logger.error("搜索引擎未初始化")
        raise HTTPException(status_code=500, detail="搜索引擎未初始化")

    try:
        logger.info(f"收到聊天完成请求: {request}")
        prompt = request.messages[-1].content
        logger.info(f"处理提示: {prompt}")

        if request.model == "graphrag-global-search:latest":
            result = await global_search_engine.asearch(prompt)
            formatted_response = format_response(result.response)
        elif request.model == "full-model:latest":
            formatted_response = await full_model_search(prompt)
        elif request.model == "graphrag-local-search:latest":
            result = await local_search_engine.asearch(prompt)
            formatted_response = format_response(result.response)

        logger.info(f"格式化的搜索结果:\n {formatted_response}")

        if request.stream:

            async def generate_stream():
                chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
                lines = formatted_response.split("\n")
                for i, line in enumerate(lines):
                    chunk = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": line + "\n"},
                                "finish_reason": None,
                            }
                        ],
                    }

                    yield f"data: {json.dumps(chunk)}\n"
                    await asyncio.sleep(0.5)
                final_chunk = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(final_chunk)}\n"
                yield "data: [DONE]\n"

            return StreamingResponse(generate_stream(), media_type="text/event-stream")

        else:
            response = ChatCompletionResponse(
                model=request.model,
                choices=[
                    ChatCompletionResponseChoice(
                        index=0,
                        message=Message(role="assistant", content=formatted_response),
                        finish_reason="stop",
                    )
                ],
                usage=Usage(
                    prompt_tokens=len(prompt.split()),
                    completion_tokens=len(formatted_response.split()),
                    total_tokens=len(prompt.split()) + len(formatted_response.split()),
                ),
            )
            logger.info(f"发送响应: \n\n{response}")
            return JSONResponse(content=response.dict())

    except Exception as e:
        logger.error(f"处理聊天完成时出错:\n\n {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
async def list_models():
    logger.info("收到模型列表请求")
    current_time = int(time.time())
    models = [
        {
            "id": "graphrag-local-search:latest",
            "object": "model",
            "created": current_time - 100000,
            "owned_by": "graphrag",
        },
        {
            "id": "graphrag-global-search:latest",
            "object": "model",
            "created": current_time - 95000,
            "owned_by": "graphrag",
        },
        {
            "id": "full-model:latest",
            "object": "model",
            "created": current_time - 80000,
            "owned_by": "combined",
        },
    ]

    response = {"object": "list", "data": models}

    logger.info(f"发送模型列表: {response}")
    return JSONResponse(content=response)


if __name__ == "__main__":
    logger.info(f"在端口 {PORT} 上启动服务器")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
