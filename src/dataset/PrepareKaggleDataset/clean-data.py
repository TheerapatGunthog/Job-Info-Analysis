import pandas as pd
import re
from pathlib import Path
from nltk.corpus import stopwords

# Load data
EXTERNAL_DATA_DIR = Path("../../../data/external")
df = pd.read_csv(EXTERNAL_DATA_DIR / "filtered_job_descriptions.csv")


# Clean data
def clean_text(text):
    if isinstance(text, float):
        return ""
    # 1. remove html tags
    text = re.sub(r"<.*?>", "", text)
    # 2. remove non-alphanumeric characters
    text = re.sub(r"[^a-zA-Z0-9.,'\" ]+", " ", text)
    # 3. make text lowercase
    text = text.lower()
    # 4. remove stopwords
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Apply the clean_text function to the description column
df["description"] = df["description"].astype(str).apply(clean_text)

output_path = EXTERNAL_DATA_DIR / "cleaned_data.csv"
df.to_csv(output_path, index=False)
print("successfully cleaned data and saved to", output_path)
