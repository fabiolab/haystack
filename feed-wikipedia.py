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
import json
from pathlib import Path
from typing import Dict, List

import typer
from langdetect import detect

from haystack import Document
from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack.nodes import DensePassageRetriever, QuestionGenerator
from haystack.nodes.file_converter.pdf import PDFToTextConverter
from haystack.nodes.file_converter.docx import DocxToTextConverter
from haystack.nodes.preprocessor import PreProcessor
from loguru import logger

from haystack.pipelines import QuestionGenerationPipeline

ELASTIC_HOST = "yd-deskin-demo.rd.francetelecom.fr"
app = typer.Typer()


@app.command()
def feeder(
    index: str,
    source_folder: Path,
    is_dense: bool = typer.Option(False),
    generate_questions: bool = typer.Option(False),
):
    document_store = ElasticsearchDocumentStore(
        host=ELASTIC_HOST, username="", password="", index=index, search_fields=["content", "title.lax"]
    )

    pdf_converter = PDFToTextConverter(remove_numeric_tables=True, valid_languages=["fr", "en"])
    docx_converter = DocxToTextConverter(remove_numeric_tables=True, valid_languages=["fr", "en"])

    if generate_questions:
        question_generator = QuestionGenerator(
            model_name_or_path="lincoln/barthez-squadFR-fquad-piaf-question-generation", prompt=""
        )
        # question_generator = QuestionGenerator(model_name_or_path="JDBN/t5-base-fr-qg-fquad")
        question_generation_pipeline = QuestionGenerationPipeline(question_generator)
        question_store = ElasticsearchDocumentStore(
            host=ELASTIC_HOST, username="", password="", index=f"{index}_questions"
        )

    # Ref : https://haystack.deepset.ai/components/preprocessing
    processor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=True,
        split_by="word",
        split_length=200,
        split_respect_sentence_boundary=True,
    )

    docs = []
    compteur = 0
    files_json = Path(source_folder).rglob("**/*")
    for path in files_json:
        if path.is_dir():
            continue

        if path.suffix.lower() == ".pdf":
            cur_meta = {"title": path.name, "category": "a", "name": get_info(path)}
            documents = pdf_converter.convert(path, meta=cur_meta, encoding="UTF-8")
        elif path.suffix.lower() in [".doc", ".docx"]:
            cur_meta = {"title": path.name, "category": "a", "name": get_info(path)}
            documents = docx_converter.convert(path, meta=cur_meta, encoding="UTF-8", remove_numeric_tables=True)
        elif path.suffix.lower() == ".json":
            documents = to_documents(path)

        compteur += 1
        logger.info(f"|___ {path.name}")

        # clean and split each dict (1x doc -> multiple docs)
        d = processor.process(documents)

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

    if generate_questions:
        questions = []
        for idx, doc in enumerate(document_store):
            for question in guess_questions(doc, question_generation_pipeline):
                questions.append({"content": question})

            if idx % 5 == 0:
                logger.info(f"==> Push {len(questions)} docs to elastic")
                question_store.write_documents(questions)
                questions = []

    if is_dense:
        # Retriever
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


def get_info(path: Path) -> str:
    with open(path.with_suffix(f"{path.suffix}.info")) as url:
        return url.readline()


def to_document(content: str, title: str, url: str) -> Dict:
    return {
        "content": content,
        "title": title,
        "meta": {"name": url, "category": "a"},
        "content_type": "text",
    }


def to_documents(file_path: Path) -> List[Dict]:
    with open(file_path) as file_json:
        articles = []
        for article in file_json.readlines():
            json_data = json.loads(article)
            articles.append(to_document(content=json_data["content"], title=json_data["title"], url=json_data["name"]))

        return articles


def guess_questions(document: Document, question_generation_pipeline: QuestionGenerationPipeline) -> List[str]:
    logger.debug(f"Generating questions for document {document.content[:50]}")
    document.content = document.content.replace(",", ".").replace(";", ".")
    # sentences = document.content.replace(",", ".").replace(";", ".").split(". ")
    questions = []
    # for idx, sentence in enumerate(sentences):
    #     document.content = f"{'.'.join(sentences[:idx])}<hl>{sentence}<hl>{'.'.join(sentences[idx+1:])}"
    logger.debug(document.content)
    results = question_generation_pipeline.run(documents=[document])
    for result in results["generated_questions"]:
        for question in result["questions"]:
            logger.info(f" - {question}")
            questions.append(question)

    return questions


if __name__ == "__main__":
    typer.run(feeder)
