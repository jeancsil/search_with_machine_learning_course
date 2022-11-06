import fasttext

# Then generate synonyms for these 1000 words. That’s easiest to do in python. Write code that:
# Loads the fastText model you created in the previous step (and probably stored in /workspace/datasets/fasttext/title_model.bin).
# Iterates through each line of /workspace/datasets/fasttext/top_words.txt (or wherever you stored the top 1,000 title words).
# Uses the model’s get_nearest_neighbors method to obtain each word’s nearest neighbors. Those are returned as an array of (similarity, word) pairs.
# Outputs, for each word, a comma-separated line that starts with the word and is followed by the neighbors whose similarity exceeds a threshold.
# Try setting the threshold to be 0.75 or 0.8.
# Store the comma-separated output in /workspace/datasets/fasttext/synonyms.csv
# Loads the model
model_path = "/workspace/datasets/fasttext/title_model2.bin"
print("Loading the model located in {}".format(model_path))
model = fasttext.load_model(model_path)

top_words_file = "/workspace/datasets/fasttext/top_words.txt"
print("Iterating over {}".format(top_words_file))

output_csv = "/workspace/datasets/fasttext/synonyms.csv"

threshold = 0.8
output_rows = []
with open(top_words_file) as top_words:
    for top_word in top_words.readline():
        synonyms = model.get_nearest_neighbors(top_word)

        row = [top_word]
        for synonym in synonyms:
            syn_score = synonym[0]
            syn_word = synonym[1]

            if syn_score < threshold:
                continue
            row.append(syn_word)
            if len(row) > 1:
                output_rows.append(",".join(row))

with open(output_csv, 'w') as o:
    for r in output_rows:
        o.write(r)
