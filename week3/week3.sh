#!/usr/bin/env bash

python create_labeled_queries.py
wc /workspace/datasets/fasttext/labeled_queries.txt
head /workspace/datasets/fasttext/labeled_queries.txt
cut -d' ' -f1 /workspace/datasets/fasttext/labeled_queries.txt | sort | uniq | wc

### Shuffle the whole labeled dataset
shuf /workspace/datasets/fasttext/labeled_queries.txt --random-source=<(seq 999999) >/workspace/datasets/fasttext/shuffled_labeled_queries.txt
### Split into training and testing
head -n 25000 /workspace/datasets/fasttext/shuffled_labeled_queries.txt >/workspace/datasets/fasttext/week3_training_data.txt
tail -n 25000 /workspace/datasets/fasttext/shuffled_labeled_queries.txt >/workspace/datasets/fasttext/week3_testing_data.txt

### Train the model
~/fastText-0.9.2/fasttext supervised -input /workspace/datasets/fasttext/week3_training_data.txt -output /workspace/search_with_machine_learning_course/week3/query_category_model -epoch 25 -lr 0.5
### Test the model
~/fastText-0.9.2/fasttext test /workspace/search_with_machine_learning_course/week3/query_category_model.bin /workspace/datasets/fasttext/week3_testing_data.txt 5
N       25000
P@5     0.147
R@5     0.733

### Train the model
~/fastText-0.9.2/fasttext supervised -input /workspace/datasets/fasttext/week3_training_data.txt -output /workspace/search_with_machine_learning_course/week3/query_category_model -epoch 35 -lr 0.5
### Test the model
~/fastText-0.9.2/fasttext test /workspace/search_with_machine_learning_course/week3/query_category_model.bin /workspace/datasets/fasttext/week3_testing_data.txt 5
N       25000
P@5     0.146
R@5     0.732

### Split AGAIN into training and testing
head -n 50000 /workspace/datasets/fasttext/shuffled_labeled_queries.txt >/workspace/datasets/fasttext/week3_training_data.txt
tail -n 50000 /workspace/datasets/fasttext/shuffled_labeled_queries.txt >/workspace/datasets/fasttext/week3_testing_data.txt

### Train the model
~/fastText-0.9.2/fasttext supervised -input /workspace/datasets/fasttext/week3_training_data.txt -output /workspace/search_with_machine_learning_course/week3/query_category_model -epoch 25 -lr 0.5
### Test the model
~/fastText-0.9.2/fasttext test /workspace/search_with_machine_learning_course/week3/query_category_model.bin /workspace/datasets/fasttext/week3_testing_data.txt 5
N       50000
P@5     0.154
R@5     0.77

### Bad classification:
#Query: football
#Response: cat02015 (Best Buy > Movies & Music > Movies & TV Shows)
