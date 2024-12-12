# Improving Long-Text Retrieval with GraphRAG: Coreference Resolution Optimization and Evaluation

## Environment Configuration

In addition to the conda environment configurations for GraphRAG and OpenCompass, you also need to install the following dependencies in order to use API profiling:

`pip install "opencompass[api]"`


To successfully configure the OpenCompass environment, in addition to the necessary installation of OpenCompass itself, you also need to install support for `data` and `api`. For these contents, please refer to the [documentation](https://github.com/open-compass/opencompass) of OpenCompass.

---

## Start and configure

To run GraphRAG's evaluation normally, you first need to start one-api and configure the API in the GraphRAG folder.

### GraphRAG folder configuration

- **`.env` file**  
  Need to replace`GRAPHRAG_CHAT_API_KEY`and`GRAPHRAG_EMBEDDING_API_KEY`with actual API key.

- **`settings.yaml` file**  
  The parameters in this file have been adjusted for the Zhipu model. If you use other models, you need to adjust them accordingly.

- **`utils/main.py` file**  
  - Need to replace the API Key in the `setup_llm_and_embedder` file to the actual API key.
  - Adjust the path to your local path`GraphRAGTest/ragtest/inputs/artifacts`（line 48 ）.
  - Make sure the API key in lines 110 and 119 has been replaced with the converted API key from OneAPI.

### OpenCompass folder configuration

- Run `run.sh` to start evaluating, `eval_api_zhipu_v2.py`is the genearal testing with the Zhipu model.
  If you need to use this evaluation, you need to modify the API key in `opencompass/configs/api_examples/eval_api_zhipu_v2.py`.

- Running `configs/eval_myTest.py`will start evaluation with GraphRAG，but need to `one-api` service as prior step.

---

## GraphRAG Evaluation File Description

To conduct the evaluation of GraphRAG, the primary file is `opencompass/opencompass/models/GraphRAGModel.py`, which includes the following functionalities:

- **Code for splitting the prompt into knowledge and question parts**  
  This splits the input prompt into a knowledge section and a question section to better organize retrieval.

- **Code for storing the extracted knowledge part into GraphRAG**  
  Ensures the extracted knowledge is effectively stored for retrieval purposes.

- **Code for running the indexing process**  
  Currently configured to retry up to 7 times to ensure the indexing process succeeds, as indexing might fail.
  **Note**：You need to reconfigure the indexing command path in `_run_indexing_command()`.

- **Code for starting the GraphRAG API to receive questions**  
  This part of the code launches an additional command window to wait for question inputs. 
  **Note**：You need to reconfigure `api_command` in`_start_api_service()`.

- **Code for sending questions and receiving answers**  
  Sends questions to the server via the API and receives results returned by the model.

- **Code for shutting down the GraphRAG API**  
  Ensures that the service is shut down after the testing ends.



