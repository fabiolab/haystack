# -------------------------------  Copyright ---------------------------------
# This software has been developed by Orange TGI/DATA-IA/AITT/aRod
# Copyright (c) Orange TGI Data/IA 2021
#
# COPYRIGHT: This file is the property of Orange TGI.
# Copyright Orange TGI 2021, All Rights Reserved
#
# This software is the confidential and proprietary information of Orange TGI.
#
# You shall not disclose such Confidential Information and shall use it only
# in accordance with the terms of the license agreement you entered into with
# Orange TGI.
#
# AUTHORS      : Orange TGI  TGI/DATA-IA/AITT/aRod
# EMAIL        : data-ia.aitt-dev-rennes@orange.com
# ----------------------------------------------------------------------------
import glob
import json
import logging
import os
from enum import Enum
from pathlib import Path
from typing import Dict, List

import typer

from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack.nodes import DensePassageRetriever
from haystack.nodes.file_converter.pdf import PDFToTextConverter
from haystack.nodes.preprocessor import PreProcessor
from loguru import logger


ELASTIC_HOST = "yd-deskin-demo.rd.francetelecom.fr"
SOURCE_FOLDER = "data/wikipedia/jo"
app = typer.Typer()

INDEX_NAME = "wikipedia"

@app.command()
def feeder(clear_index: bool = typer.Option(False)):

    # ## Document Store
    # Connect to Elasticsearch
    document_store = ElasticsearchDocumentStore(host=ELASTIC_HOST, username="", password="", index=INDEX_NAME)

    if clear_index:
        document_store.delete_documents(index=INDEX_NAME)

    # ## Preprocessing of documents
    converter = PDFToTextConverter(remove_numeric_tables=True, valid_languages=["fr", "en"])

    # TODO: Vérifier les paramètres de préprocessing. Split length => 200 ? Split_by "passage" ?
    # https://haystack.deepset.ai/components/preprocessing
    processor = PreProcessor(clean_empty_lines=True,
                             clean_whitespace=True,
                             clean_header_footer=True,
                             split_by="word",
                             split_length=200,
                             split_respect_sentence_boundary=True)

    docs = []
    compteur = 0
    files_json = Path(SOURCE_FOLDER).rglob("**/*")
    for path in files_json:
        if not path.is_dir():
            documents = to_documents(path)
            compteur += 1

            logger.info(f"|___ {path.name}")

            # clean and split each dict (1x doc -> multiple docs)
            d = processor.process(documents)
            docs.extend(d)

            if compteur % 100 == 0:
                logger.info("Indexing docs in elastic")
                document_store.write_documents(docs)
                docs = []

    # Now, let's write the docs to our DB.
    document_store.write_documents(docs)

    ## Retriever
    retriever = DensePassageRetriever(document_store=document_store,
                                      query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
                                      passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
                                      max_seq_len_query=64,
                                      max_seq_len_passage=200,
                                      batch_size=2,
                                      use_gpu=True,
                                      embed_title=True,
                                      use_fast_tokenizers=True
                                      )

    # Important:
    # Now that after we have the DPR initialized, we need to call update_embeddings() to iterate over all
    # previously indexed documents and update their embedding representation.
    # While this can be a time consuming operation (depending on corpus size), it only needs to be done once.
    # At query time, we only need to embed the query and compare it the existing doc embeddings which is very fast.
    document_store.update_embeddings(retriever)


def to_documents(file_path: Path) -> List[Dict]:
    with open(file_path) as file_json:
        articles = [{"content":json.loads(article)['text'], "meta": {"name": json.loads(article)['url'], "category": "a"}, "content_type":"text"} for article in file_json.readlines()]
        return articles


if __name__ == "__main__":
    typer.run(feeder)

