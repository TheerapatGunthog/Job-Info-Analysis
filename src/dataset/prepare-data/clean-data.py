import pandas as pd
import re
from pathlib import Path
from nltk.corpus import stopwords
import nltk

# Ensure NLTK stopwords are downloaded
nltk.download("stopwords")

# Define the directory and load data
INTERIM_DATA_DIR = Path("/home/whilebell/Code/Project/Job-Info-Analysis/data/interim")
df = pd.read_csv(INTERIM_DATA_DIR / "filtered_job_descriptions.csv")

# Compile regex patterns for performance
html_tags_pattern = re.compile(r"<.*?>")
non_alphanumeric_pattern = re.compile(r"[^a-zA-Z0-9.'\" ]+")
sentence_end_pattern = re.compile(r"(?<!\w\.)[.,](?!\s|$)")

# Load stopwords
stop_words = set(stopwords.words("english"))


# Clean data
def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Remove HTML tags
    text = html_tags_pattern.sub("", text)
    # Remove non-alphanumeric characters except sentence-ending dots and commas
    text = non_alphanumeric_pattern.sub(" ", text)
    # Remove dots and commas within sentences
    text = sentence_end_pattern.sub("", text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove stopwords
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text


# Apply the clean_text function to the description column
df["description"] = df["description"].astype(str).apply(clean_text)

# Log missing descriptions
missing_descriptions = df["description"].isna().sum()
print(f"จำนวนข้อมูลที่ไม่มีคำอธิบาย: {missing_descriptions}")

# Save cleaned data
output_path = INTERIM_DATA_DIR / "cleaned_data.csv"
df.to_csv(output_path, index=False)

# Logging success
print(f"Successfully cleaned data and saved to {output_path}")
