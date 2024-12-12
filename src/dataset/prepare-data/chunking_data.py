import spacy
from transformers import BertTokenizer
import pandas as pd
from pathlib import Path
from tqdm import tqdm

tqdm.pandas()

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load("en_core_web_sm")

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("distilbert-base-uncased")


# Function to split text into sentences
def split_sentences(text):
    doc = nlp(text)
    return [sent.text for sent in doc.sents]


# Function to chunk sentences into chunks of max_tokens
def chunk_sentences(sentences, max_tokens=50):
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(tokenizer.tokenize(sentence))

        # If adding the sentence to the current chunk would exceed the maximum token length
        if current_length + sentence_length > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0

        # Add the sentence to the current chunk
        current_chunk.append(sentence)
        current_length += sentence_length

    # Add the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    # If no chunking occurred, return the original sentences
    if len(chunks) == 1 and len(sentences) == 1:
        return sentences

    return chunks


INTERIM_DATA_DIR = Path("/home/whilebell/Code/Project/Job-Info-Analysis/data/interim")
df = pd.read_csv(INTERIM_DATA_DIR / "cleaned_data.csv")

# Debugging NaN value in dataset
df = df.dropna(subset=["description"])

# Apply the chunking process to the description column
df["chunks"] = df["description"].progress_apply(
    lambda x: chunk_sentences(split_sentences(x))
)

# Show df number of rows
print(f"Number of rows: {df.shape[0]}")

# Save the chunked data to a new CSV file
output_path = INTERIM_DATA_DIR / "chunking_data.csv"
df.to_csv(output_path, index=False)
print(f"Successfully saved chunked data to {output_path}")
