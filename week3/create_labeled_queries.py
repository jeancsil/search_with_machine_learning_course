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

categories_file_name = r'/tmp/workspace/datasets/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml'

queries_file_name = r'/tmp/workspace/datasets/train.csv'
output_file_name = r'/tmp/workspace/datasets/fasttext/labeled_queries.txt'

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


min_queries = 1
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
## If a category is associated with less than the minimum number of queries, then it gets rolled up to its parent category.
## Loop queries_df(category, query) and if a category has too few queries
## Get the parent of this category and see if the above is false

queries_df['query'] = queries_df['query'].apply(lambda x: fix_normalization(x))
# -----------------------------
queries_df_with_counts = queries_df.groupby('category').size().reset_index(name='counts')
queries_with_parent_df = queries_df.merge(queries_df_with_counts, how='left', on='category').merge(parents_df,
                                                                                                   how='left',
                                                                                                   on='category')

while 0 < len(queries_with_parent_df[queries_with_parent_df['counts'] < min_queries]):
    queries_with_parent_df.loc[queries_with_parent_df['counts'] < min_queries, 'category'] = queries_with_parent_df[
        'parent']
    queries_df = queries_with_parent_df[['category', 'query']]
    queries_df = queries_df[queries_df['category'].isin(categories)]
    queries_df_with_counts = queries_df.groupby('category').size().reset_index(name='counts')
    queries_with_parent_df = queries_df.merge(queries_df_with_counts, how='left', on='category').merge(parents_df,
                                                                                                       how='left',
                                                                                                       on='category')
# -----------------------------
# queries_with_count_df = queries_df.groupby(['category']).size().reset_index(name='count')
# # print(queries_with_count_df)
#
# categories_without_enough_queries = queries_with_count_df[queries_with_count_df['count'] < min_queries]
# print("Categories without enough queries: {}".format(len(categories_without_enough_queries)))
# # print("========")
# for idx, row in categories_without_enough_queries.iterrows():
#     has_enough_queries = False
#     while not has_enough_queries:
#         print("looking for " + row['category'])
#         found = parents_df[parents_df['category'] == row['category']]['parent']
#         if len(found) <= 0:
#             print("NOT FOUND")
#             exit(1)
#         parent_category = parents_df.loc[found.index]
#         print("\n")
#         print(parent_category['parent'].values[0])
#         print("\n")
#         print(queries_with_count_df[queries_with_count_df['category'] == parent_category['parent'].values[0]])
#         # print(queries_with_count_df[queries_with_count_df['category'] == parent_category['parent']])
#         # print(queries_with_count_df['category'] == parent_category)
#         # parent_category_count = queries_with_count_df[queries_with_count_df['category'] == parent_category]
#         # print(parent_category_count)
#         exit(0)
#         #has_enough_queries = parent_category_count >= min_queries
#         # print(found.index)
#         print()
#         # exit(0)
#         # parent_category = parents_df[parents_df['category'] == row['category']]['parent']
#         # print(queries_with_count_df[queries_with_count_df['category'] == parent_category])
#         # has_enough_queries = queries_with_count_df[queries_with_count_df['category'] == parent_category][
#         #                         'count'] >= min_queries
#         # print("=====> updating the count of categories to " + queries_with_count_df[queries_with_count_df['category'] == parent_category][
#         #                         'count'] + " results in " + has_enough_queries)
#         # print(row['category'], "parent:", parent_category)
#         # break
#

#
# print(queries_df.head(10))
# print(parents_df.head(10))

# lixo = len(queries_df[queries_df["category"] == "abcat0701001"])
# print(lixo == 13830)

print(len(queries_df[queries_df["category"] == "abcat0701001"]) == 13830)
print(len(queries_with_parent_df[queries_with_parent_df['counts'] >= min_queries]))


# Create labels in fastText format.
queries_df['label'] = '__label__' + queries_df['category']

# Output labeled query data as a space-separated file, making sure that every category is in the taxonomy.
queries_df = queries_df[queries_df['category'].isin(categories)]
queries_df['output'] = queries_df['label'] + ' ' + queries_df['query']
queries_df[['output']].to_csv(output_file_name, header=False, sep='|', escapechar='\\', quoting=csv.QUOTE_NONE,
                              index=False)
