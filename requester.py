# -------------------------------  Copyright ---------------------------------
# This software has been developed by Orange TGI/DATA-IA/AITT/aRod
# Copyright (c) Orange TGI Data/IA 2022
#
# COPYRIGHT: This file is the property of Orange TGI.
# Copyright Orange TGI 2022, All Rights Reserved
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
from pprint import pprint
from haystack.pipelines import ExtractiveQAPipeline
from haystack.nodes import DensePassageRetriever, ElasticsearchRetriever, FARMReader
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.utils import print_answers

ELASTIC_HOST = "yd-deskin-demo.rd.francetelecom.fr"
INDEX_NAME = "wikipedia"

document_store = ElasticsearchDocumentStore(host=ELASTIC_HOST, username="", password="", index=INDEX_NAME)

retriever = DensePassageRetriever(document_store=document_store,
                                      query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
                                      passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
                                      max_seq_len_query=64,
                                      max_seq_len_passage=256,
                                      batch_size=16,
                                      use_gpu=True,
                                      embed_title=True,
                                      use_fast_tokenizers=True
                                      )

reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)

pipe = ExtractiveQAPipeline(reader, retriever)

if __name__=="__main__":
    while True:
        query = input("Type your query:")
        prediction = pipe.run(
            query=query, params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}}
        )

        pprint(prediction)
        print_answers(prediction, details="minimum")
