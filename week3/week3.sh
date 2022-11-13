#!/usr/bin/env bash

python create_labeled_queries.py
wc /workspace/datasets/fasttext/labeled_queries.txt
head /workspace/datasets/fasttext/labeled_queries.txt
cut -d' ' -f1 /workspace/datasets/fasttext/labeled_queries.txt | sort | uniq | wc