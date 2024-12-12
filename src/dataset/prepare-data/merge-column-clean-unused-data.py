import pandas as pd
import yaml
import re
from pathlib import Path

# Create directories path
EXTERNAL_DATA_DIR = Path("/home/whilebell/Code/Project/Job-Info-Analysis/data/external")
RAW_DATA_DIR = Path("/home/whilebell/Code/Project/Job-Info-Analysis/data/raw")
INTERIM_DATA_DIR = Path("/home/whilebell/Code/Project/Job-Info-Analysis/data/interim")

dataset1 = EXTERNAL_DATA_DIR / "job-description-dataset/job_descriptions.csv"
dataset2 = EXTERNAL_DATA_DIR / "jobs-dataset-from-glassdoor/eda_data.csv"
dataset3 = EXTERNAL_DATA_DIR / "LinkedIn Job Postings (2023 - 2024)/postings.csv"
dataset4 = (
    EXTERNAL_DATA_DIR / "U.S. Technology Jobs on Dice.com/dice_com-job_us_sample.csv"
)
dataset5 = EXTERNAL_DATA_DIR / "jobs-and-job-description/job_title_des.csv"
dataset6 = EXTERNAL_DATA_DIR / "jobs.csv"


# Load keywords from yaml file
def load_keywords(yaml_file):
    with open(yaml_file, "r") as file:
        data = yaml.safe_load(file)
    return data.get("keywords", [])


# Load keywords from yaml file
yaml_file = RAW_DATA_DIR / "jobs-title-keyword.yaml"
keywords = load_keywords(yaml_file)


def escape_regex(pattern):
    return re.escape(pattern).replace(r"\ ", " ")


# Filter jobs based on keywords in title
def filter_jobs(file_path, title_col, description_col):
    global all_descriptions  # To access all_descriptions DataFrame
    try:
        # Load data
        df = pd.read_csv(file_path)
        df = df.dropna(subset=[description_col])
        # Check if the columns of interest exist in the file
        if title_col in df.columns and description_col in df.columns:
            # Drop rows with NaN in the title column
            df = df.dropna(subset=[title_col])
            # Convert title column to string and lowercase
            df[title_col] = df[title_col].astype(str).str.lower()
            # Escape special regex characters in keywords
            escaped_keywords = [escape_regex(keyword) for keyword in keywords]
            # Filter data based on keywords in title
            filtered_df = df[
                df[title_col].str.contains(
                    "|".join(escaped_keywords), case=False, na=False, regex=True
                )
            ]
            # Add filtered descriptions to all_descriptions
            all_descriptions = pd.concat(
                [
                    all_descriptions,
                    filtered_df[[description_col]].rename(
                        columns={description_col: "description"}
                    ),
                ],
                ignore_index=True,
            )
        else:
            print(
                f"Columns {title_col} or {description_col} not found in file {file_path.name}"
            )
    except Exception as e:
        print(f"Error reading file {file_path.name}: {e}")


# Create new dataframe to store all job descriptions
all_descriptions = pd.DataFrame(columns=["description"])

filter_jobs(dataset1, "Job Title", "Job Description")
filter_jobs(dataset2, "Job Title", "Job Description")
filter_jobs(dataset3, "title", "description")
filter_jobs(dataset5, "Job Title", "Job Description")

# Add a linked in only tech jobs dataset to the filtered jobs dataset
linked_tech_job_data = pd.read_csv(dataset6)
linked_tech_job_data = linked_tech_job_data[["description"]].rename(
    columns={"description": "description"}
)
all_descriptions = pd.concat(
    [all_descriptions, linked_tech_job_data], ignore_index=True
)

# Add US tech job data
us_tech_job_data = pd.read_csv(dataset4)
us_tech_job_data = us_tech_job_data[["jobdescription"]].rename(
    columns={"jobdescription": "description"}
)
all_descriptions = pd.concat([all_descriptions, us_tech_job_data], ignore_index=True)

# Remove duplicate rows
all_descriptions = all_descriptions.drop_duplicates()

# Drop rows with NaN values in the description column
all_descriptions = all_descriptions.dropna(subset=["description"])

# Check for NaN values
print("Check NaN Values:")
print(all_descriptions.isnull().sum())

# Save the filtered job descriptions to a csv file
output_path = INTERIM_DATA_DIR / "filtered_job_descriptions.csv"
all_descriptions.to_csv(output_path, index=False)
print(f"Saved filtered descriptions to file: {output_path}")
