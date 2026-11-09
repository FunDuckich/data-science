ARG BASE_IMAGE=quay.io/jupyter/scipy-notebook:python-3.11
FROM ${BASE_IMAGE}

RUN pip install \
    openai \
    httpx \
    python-dotenv datasets==3.6.0 pydantic==2.12.4 ollama==0.6.1
