import spacy
from transformers import BertTokenizer
import pandas as pd
from pathlib import Path
from tqdm import tqdm

tqdm.pandas()

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load("en_core_web_sm")

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# Function to split text into sentences
def split_sentences(text):
    doc = nlp(text)
    return [sent.text for sent in doc.sents]


# Function to chunk sentences into chunks of max_tokens
def chunk_sentences(sentences, max_tokens=512):
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

    return chunks


EXTERNAL_DATA_DIR = Path("../../../data/external")
df = pd.read_csv(EXTERNAL_DATA_DIR / "cleaned_data.csv")

# Split chunks into sentences
df["sentences"] = df["description"].progress_apply(lambda x: split_sentences(x))

# Chunk sentences into chunks of max_tokens
df["chunks"] = df["sentences"].progress_apply(
    lambda x: chunk_sentences(x, max_tokens=510)
)

# Explode chunks into separate rows
df_exploded = df.explode("chunks").reset_index(drop=True)

# Save data
output_path = EXTERNAL_DATA_DIR / "bert_ready_data.csv"
df_exploded.to_csv(output_path, index=False)

print(f"Data saved to {output_path}")
