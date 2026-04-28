"""Sanity checks for RAG environment dependencies."""

import importlib


def test_langchain_import() -> None:
    module = importlib.import_module("langchain")
    assert module is not None


def test_mistral_sdk_import() -> None:
    module = importlib.import_module("mistralai")
    assert module is not None


def test_faiss_cpu_import_and_backend() -> None:
    faiss = importlib.import_module("faiss")
    assert faiss is not None
    # faiss-cpu does not have get_num_gpus(); this function only exists in GPU builds.
