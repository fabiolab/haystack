import os
from pathlib import Path


PIPELINE_YAML_PATH = os.getenv("PIPELINE_YAML_PATH", str((Path(__file__).parent / "pipeline" / "pipelines.yaml").absolute()))
PIPELINE_DENSE_YAML_PATH = os.getenv("PIPELINE_DENSE_YAML_PATH", str((Path(__file__).parent / "pipeline" / "pipelines_dense.yaml").absolute()))
PIPELINE_WIKIPEDIA_YAML_PATH = os.getenv("PIPELINE_WIKIPEDIA_YAML_PATH", str((Path(__file__).parent / "pipeline" / "pipelines_wikipedia.yaml").absolute()))
PIPELINE_JO_YAML_PATH = os.getenv("PIPELINE_JO_YAML_PATH", str((Path(__file__).parent / "pipeline" / "pipelines_jo.yaml").absolute()))

QUERY_PIPELINE_NAME = os.getenv("QUERY_PIPELINE_NAME", "query")
QUERY_PIPELINE_DENSE_NAME = os.getenv("QUERY_PIPELINE_DENSE_NAME", "query_dense")
QUERY_PIPELINE_JO_NAME = os.getenv("QUERY_PIPELINE_JO_NAME", "query_wikipedia")

INDEXING_PIPELINE_NAME = os.getenv("INDEXING_PIPELINE_NAME", "indexing")

FILE_UPLOAD_PATH = os.getenv("FILE_UPLOAD_PATH", str((Path(__file__).parent / "file-upload").absolute()))

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
ROOT_PATH = os.getenv("ROOT_PATH", "/")

CONCURRENT_REQUEST_PER_WORKER = int(os.getenv("CONCURRENT_REQUEST_PER_WORKER", "4"))

PIPELINES_PATH = os.getenv("PIPELINE_YAML_PATH", str((Path(__file__).parent / "pipeline").absolute()))