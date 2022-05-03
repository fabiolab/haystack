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
import logging
import os
from enum import Enum

import typer

from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack.nodes import DensePassageRetriever
from haystack.nodes.file_converter.pdf import PDFToTextConverter
from haystack.nodes.preprocessor import PreProcessor
from loguru import logger
from langdetect import detect


ELASTIC_HOST = "yd-deskin-demo.rd.francetelecom.fr"
SOURCE_FOLDER = "data"
app = typer.Typer()

INDEX_NAME = "dense"


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
    processor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=True,
        split_by="word",
        split_length=200,
        split_respect_sentence_boundary=True,
    )

    docs = []

    files_pdf = glob.glob(os.path.join(SOURCE_FOLDER, "**/*.pdf"), recursive=True)

    compteur = 0
    for file in files_pdf:
        compteur += 1
        logger.info(f"|___ {file}")
        # Optional: Supply any meta data here
        # the "name" field will be used by DPR if embed_title=True, rest is custom and can be named arbitrarily
        cur_meta = {"name": file, "category": "a"}

        d = converter.convert(file, meta=cur_meta, encoding="UTF-8")

        # clean and split each dict (1x doc -> multiple docs)
        d = processor.process(d)

        multilanguage_d = []
        for single_doc in d:
            try:
                if detect(single_doc["content"]) == "en":
                    single_doc["content_en"] = single_doc["content"]
            except Exception as e:
                logger.error(f"Can't get language from {single_doc['content']}")

            multilanguage_d.append(single_doc)

        docs.extend(multilanguage_d)

        if compteur % 100 == 0:
            logger.info("Indexing docs in elastic")
            document_store.write_documents(docs)
            docs = []

    # Now, let's write the docs to our DB.
    document_store.write_documents(docs)

    ### Retriever
    retriever = DensePassageRetriever(
        document_store=document_store,
        query_embedding_model="voidful/dpr-question_encoder-bert-base-multilingual",
        passage_embedding_model="voidful/dpr-ctx_encoder-bert-base-multilingual",
        # query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
        # passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
        max_seq_len_query=64,
        max_seq_len_passage=256,
        batch_size=16,
        use_gpu=True,
        embed_title=True,
        use_fast_tokenizers=True,
    )

    # Important:
    # Now that after we have the DPR initialized, we need to call update_embeddings() to iterate over all
    # previously indexed documents and update their embedding representation.
    # While this can be a time consuming operation (depending on corpus size), it only needs to be done once.
    # At query time, we only need to embed the query and compare it the existing doc embeddings which is very fast.
    document_store.update_embeddings(retriever)


if __name__ == "__main__":
    typer.run(feeder)
