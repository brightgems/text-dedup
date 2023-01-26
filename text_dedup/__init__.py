#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2023-01-26 16:02
# @Author  : George (brightgems@gmail.com)
"""Text deduplication simplified."""

from typing import List
from .embedders.minhash import MinHashEmbedder
from .postprocess.clustering import lsh_clustering
from .postprocess.group import get_group_indices


class MinHashDeduper:
    """ Golden Minhash algorithm, compapible with v0.0.11 """
    def __init__(self, num_perm: int = 128, threshold: float = 0.5, ngram_size: int = 5):
    
        self.num_perm = num_perm
        self.threshold = threshold
        self.ngram_size = ngram_size
    
    def fit_transform(self, data: List[str]) -> List[int]:
        """Group similar documents with minhash.
        Parameters
        ----------
        data : List[str]
            List of document strings.
        Returns
        -------
        List[int]
            List of group indices.
        
        Examples
        --------
        >>> deduper = MinHashDeduper(ngram_size=5, threshold=0.3)
        >>> groups = deduper.fit_transform(["This is a sentence.", "This is another sentence.", "This is a question.", "hello world"])
        >>> groups
        [0, 0, 2, 3]
        """
        embedder = MinHashEmbedder()
        embeddings = embedder.embed(data, n_gram=self.ngram_size,level='char')

        clusters = lsh_clustering(embeddings,threshold=0.5,num_perm=128)
        return get_group_indices(clusters)
