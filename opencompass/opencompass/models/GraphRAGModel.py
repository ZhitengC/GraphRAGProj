import os
import re
import requests
import subprocess
import time
import json
from typing import List, Union
from opencompass.utils.prompt import PromptList
from opencompass.models import BaseModel
from opencompass.registry import MODELS

PromptType = Union[PromptList, str]


@MODELS.register_module()
class GraphRAGModel(BaseModel):
    def __init__(self, api_url: str, input_dir: str, retry: int = 2, **kwargs):
        self.api_url = api_url
        self.input_dir = input_dir
        self.retry = retry
        self.template_parser = self
        os.makedirs(self.input_dir, exist_ok=True)

    def parse_template(self, prompt_template, mode="gen"):
        return prompt_template

    def generate(self, inputs: List[PromptType], max_out_len: int = 512) -> List[str]:
        results = []
        for input_text in inputs:
            result = self._generate(input_text, max_out_len)
            results.append(result)
        return results

    def _generate(self, input_text: PromptType, max_out_len: int = 512) -> str:
        knowledge_content, question_content = self._extract_prompt_and_question(
            input_text
        )
        self._save_prompt(knowledge_content)

        max_retries = 7  # 当构建索引时， 有可能会因为网络连接或者其他问题失败， 这里设置最大重试次数
        retry_count = 0
        success = False

        while not success and retry_count < max_retries:
            success = self._run_indexing_command()
            if not success:
                retry_count += 1
                print(f"索引构建失败，重试 {retry_count}/{max_retries}...")
                time.sleep(2)

        if not success:
            return "Error: 索引构建失败"

        # 启动 API 服务
        api_process = self._start_api_service()
        if api_process is None:
            return "Error: API 服务启动失败"

        time.sleep(5)

        # 发送问题并获取答案
        answer = self._call_graphrag_api(question_content)

        # 关闭 API 服务
        self._stop_api_service()

        return answer

    def _extract_prompt_and_question(self, input_text: PromptType) -> (str, str):
        if isinstance(input_text, str):
            prompt_content = input_text
        else:
            prompt_content = self._extract_nested_prompt(input_text)

        knowledge_content = re.split(r"\nQ:", prompt_content)[0]
        question_match = re.search(r"\nQ:(.*)", prompt_content)
        question_content = question_match.group(1).strip() if question_match else ""

        return knowledge_content, question_content

    def _extract_nested_prompt(self, items):
        prompts = []
        for item in items:
            if isinstance(item, dict):
                if "prompt" in item and item["prompt"].strip():  # 只提取非空 prompt
                    prompts.append(item["prompt"])
                elif isinstance(item.get("origin_prompt"), list):
                    prompts.append(self._extract_nested_prompt(item["origin_prompt"]))
        return "".join(prompts)

    def _save_prompt(self, knowledge_content: str):
        prompt_path = os.path.join(self.input_dir, "currentPrompt.txt")
        with open(prompt_path, "w") as f:
            f.write(knowledge_content)

    def _run_indexing_command(self):
        # 这部分需要将以下命令替换为本地的graphrag的索引构建命令
        index_command = (
            "cd 本地的GraphragTest/ragtest地址 && "
            "source ~/anaconda3/etc/profile.d/conda.sh && "
            "conda activate 本地的graphrag环境 && "
            "python -m graphrag.index --root ./"
        )
        try:
            # 同步执行索引构建命令，等待完成
            subprocess.run(
                index_command,
                shell=True,
                check=True,
                executable="/bin/bash",
            )
            print("索引构建已完成。")
            return True
        except subprocess.CalledProcessError as e:
            print(f"索引构建失败：{e}")
            return False

    def _start_api_service(self):
        # 这部分需要将以下命令替换为本地的graphrag的API服务启动命令
        api_command = (
            "cd 本地的GraphragTest/ragtest/utils地址 && "
            "source ~/anaconda3/etc/profile.d/conda.sh && "
            "conda activate 本地的graphrag环境 && "
            "python main.py; exec bash"
        )
        try:
            # 使用 gnome-terminal 在新窗口中启动 API 服务
            subprocess.Popen(
                ["gnome-terminal", "--", "bash", "-c", api_command],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            print("API 服务已在新终端窗口中启动。")
            return True
        except Exception as e:
            print(f"API 服务启动失败：{e}")
            return False

    def _call_graphrag_api(self, question_content: str) -> str:
        headers = {"Content-Type": "application/json"}
        data = {
            "model": "graphrag-global-search:latest",
            "messages": [{"role": "user", "content": question_content}],
            "temperature": 0.7,
        }
        print(f"发送问题：{question_content}")
        for attempt in range(self.retry):
            try:
                response = requests.post(
                    self.api_url, headers=headers, data=json.dumps(data)
                )
                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"]
                else:
                    print(f"Error: Failed with status code {response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"Attempt {attempt + 1} - Connection error: {e}")
                time.sleep(2)
        return "Error: Failed to get a response from GraphRAG"

    def _stop_api_service(self):
        try:
            # 查找运行 main.py 的进程
            result = subprocess.check_output(["pgrep", "-f", "python main.py"])
            pids = result.decode().strip().split("\n")
            for pid in pids:
                # 终止进程
                subprocess.run(["kill", "-9", pid])
            print("API 服务已停止。")
        except subprocess.CalledProcessError:
            print("未找到正在运行的 API 服务。")
