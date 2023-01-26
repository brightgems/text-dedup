#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-04-02 12:54:28
# @Author  : Chenghao Mou (mouchenghao@gmail.com)

from text_dedup.embedders.transformer import TransformerEmbedder
from text_dedup.postprocess.clustering import annoy_clustering
from text_dedup.postprocess.group import get_group_indices


def test_transformer():

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    corpus = [
        'The quick brown fox jumps over the dog',
        'The quick brown fox jumps over the corgi',
        'This is a test',
        'This is a test message',
    ]

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')

    embedder = TransformerEmbedder(tokenizer, model)
    embeddings = embedder.embed(corpus)

    clusters = annoy_clustering(embeddings, f=768)
    groups = get_group_indices(clusters)
    assert groups == [0, 0, 2, 2]
