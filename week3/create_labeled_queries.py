import os
import argparse
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import csv
import re

# Useful if you want to perform stemming.
import nltk

stemmer = nltk.stem.PorterStemmer()

categories_file_name = r'/workspace/datasets/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml'

queries_file_name = r'/workspace/datasets/train.csv'
output_file_name = r'/workspace/datasets/fasttext/labeled_queries.txt'

parser = argparse.ArgumentParser(description='Process arguments.')
general = parser.add_argument_group("general")
general.add_argument("--min_queries", default=1, help="The minimum number of queries per category label (default is 1)")
general.add_argument("--output", default=output_file_name, help="the file to output to")

args = parser.parse_args()
output_file_name = args.output


def fix_normalization(input_str: str):
    input_str = input_str.lower()
    input_str = re.sub('[^a-z0-9]', ' ', input_str)
    input_str = re.sub('\s+', ' ', input_str)
    # return input_str
    return " ".join([stemmer.stem(x) for x in input_str.split(" ")])


if args.min_queries:
    min_queries = int(args.min_queries)

# The root category, named Best Buy with id cat00000, doesn't have a parent.
root_category_id = 'cat00000'

tree = ET.parse(categories_file_name)
root = tree.getroot()

# Parse the category XML file to map each category id to its parent category id in a dataframe.
categories = []
parents = []
for child in root:
    id = child.find('id').text
    cat_path = child.find('path')
    cat_path_ids = [cat.find('id').text for cat in cat_path]
    leaf_id = cat_path_ids[-1]
    if leaf_id != root_category_id:
        categories.append(leaf_id)
        parents.append(cat_path_ids[-2])
parents_df = pd.DataFrame(list(zip(categories, parents)), columns=['category', 'parent'])

# Read the training data into pandas, only keeping queries with non-root categories in our category tree.
queries_df = pd.read_csv(queries_file_name)[['category', 'query']]
queries_df = queries_df[queries_df['category'].isin(categories)]

# IMPLEMENT ME: Convert queries to lowercase, and optionally implement other normalization, like stemming.
queries_df['query'] = queries_df['query'].apply(lambda x: fix_normalization(x))

# Counts
queries_grouped_df = queries_df.groupby(['category']).size().reset_index(name='query_count')
queries_grouped_df['should_include'] = queries_grouped_df['query_count'].apply(lambda x: True if x >= min_queries else False)
# print(len(queries_df[queries_df["category"] == "abcat0701001"]))
# print(len(queries_grouped_df[queries_grouped_df['query_count'] >= 1000]))

# IMPLEMENT ME: Roll up categories to ancestors to satisfy the minimum number of queries per category.
while ((~queries_grouped_df['should_include']).sum() > 0):
    queries_grouped_df = queries_grouped_df.merge(parents_df, how='left', on='category')
    queries_df = queries_df.merge(queries_grouped_df, how='left', on='category')

    queries_df['category'] = queries_df.apply(
        lambda x: str(x['category']) if x['should_include'] else str(x['parent']), axis=1)
    queries_df = queries_df.drop(columns=['parent', 'should_include', 'query_count'])

    queries_grouped_df = queries_df.groupby(['category']).agg(count=('query', 'query_count')).reset_index()
    queries_grouped_df['should_include'] = queries_grouped_df['query_count'].apply(lambda x: True if x >= min_queries else False)


# Create labels in fastText format.
queries_df['label'] = '__label__' + queries_df['category']

# Output labeled query data as a space-separated file, making sure that every category is in the taxonomy.
queries_df = queries_df[queries_df['category'].isin(categories)]
queries_df['output'] = queries_df['label'] + ' ' + queries_df['query']
queries_df[['output']].to_csv(output_file_name, header=False, sep='|', escapechar='\\', quoting=csv.QUOTE_NONE,
                              index=False)
