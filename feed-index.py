#
# Example of use
#
# > python feed-index.py jo data/wikipedia/jo/json/

import json
from pathlib import Path
from typing import Dict, List

import typer

from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack.nodes.preprocessor import PreProcessor
from loguru import logger


ELASTIC_HOST = "yd-deskin-demo.rd.francetelecom.fr"
app = typer.Typer()


@app.command()
def feeder(index: str, source_dir_path: str, clear_index: bool = typer.Option(False)):
    logger.info(f"Feed the index {index} on {ELASTIC_HOST} with data from {source_dir_path}")
    document_store = ElasticsearchDocumentStore(host=ELASTIC_HOST, username="", password="", index=index, search_fields=["content", "title"])
    if clear_index:
        document_store.delete_documents(index=index)

    processor = PreProcessor(clean_empty_lines=True,
                             clean_whitespace=True,
                             clean_header_footer=True,
                             split_by="word",
                             split_length=1000,
                             split_respect_sentence_boundary=True)

    docs = []
    compteur = 0
    files_json = Path(source_dir_path).rglob("**/*")
    for path in files_json:
        if not path.is_dir():
            documents = to_documents(path)
            compteur += 1
            d = processor.process(documents)
            docs.extend(d)
            if compteur % 100 == 0:
                logger.info("Indexing docs in elastic")
                document_store.write_documents(docs)
                docs = []
    document_store.write_documents(docs)


def to_documents(file_path: Path) -> List[Dict]:
    with open(file_path) as file_json:
        articles = [{"content":json.loads(article)['text'], "title": json.loads(article)['title'], "meta": {"name": json.loads(article)['url'], "category": "a"}, "content_type":"text"} for article in file_json.readlines()]
        return articles


if __name__ == "__main__":
    typer.run(feeder)

