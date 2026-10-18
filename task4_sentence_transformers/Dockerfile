ARG TORCH_BASE=pytorch
FROM quay.io/jupyter/scipy-notebook:python-3.11 AS pytorch

RUN pip install --no-cache-dir torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
FROM ${TORCH_BASE}
RUN pip install --no-cache-dir sentence-transformers==5.1.1 transformers==4.57.1 beautifulsoup4==4.14.2 pyyaml==6.0.3
