import pandas as pd
import yaml
from pathlib import Path
import re

# Create directories path
EXTERNAL_DATA_DIR = Path("../../../data/external")
RAW_DATA_DIR = Path("../../../data/raw")
INTERIM_DATA_DIR = Path("../../../data/interim")

jobdescriptiondataset = (
    EXTERNAL_DATA_DIR / "job-description-dataset/job_descriptions.csv"
)
datasetglassdoor = EXTERNAL_DATA_DIR / "jobs-dataset-from-glassdoor/eda_data.csv"
linkedinjobposting = (
    EXTERNAL_DATA_DIR / "LinkedIn Job Postings (2023 - 2024)/postings.csv"
)
ustechnologyjobsondice = (
    EXTERNAL_DATA_DIR / "U.S. Technology Jobs on Dice.com/dice_com-job_us_sample.csv"
)
jobsandjobdescription = EXTERNAL_DATA_DIR / "jobs-and-job-description/job_title_des.csv"

# Columns that contain job title and job description in each dataset
file_columns = {
    jobdescriptiondataset: ["Job Title", "Job Description"],
    datasetglassdoor: ["Job Title", "Job Description"],
    linkedinjobposting: ["title", "description"],
    ustechnologyjobsondice: ["jobtitle", "jobdescription"],
    jobsandjobdescription: ["Job Title", "Job Description"],
}


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
def filter_jobs(file_path, columns):
    try:
        # Load data
        df = pd.read_csv(file_path)
        # Check if the columns of interest exist in the file
        if all(col in df.columns for col in columns):
            # Drop rows with NaN in the title column
            df = df.dropna(subset=columns)
            # Convert title column to string and lowercase
            df[columns[0]] = df[columns[0]].astype(str).str.lower()
            # Escape special regex characters in keywords
            escaped_keywords = [escape_regex(keyword) for keyword in keywords]
            # Filter data based on keywords in title
            filtered_df = df[
                df[columns[0]].str.contains(
                    "|".join(escaped_keywords), case=False, na=False, regex=True
                )
            ]
            return filtered_df
        else:
            print(f"Columns {columns} not found in file {file_path.name}")
            return pd.DataFrame()
    except Exception as e:
        print(f"Error reading file {file_path.name}: {e}")
        return pd.DataFrame()


# Merge all datasets and filter jobs based on keywords in title
all_filtered_jobs = pd.DataFrame()
for file_path, columns in file_columns.items():
    filtered_jobs = filter_jobs(file_path, columns)
    all_filtered_jobs = pd.concat([all_filtered_jobs, filtered_jobs], ignore_index=True)

# Print the number of rows after merging and filtering
print(f"Number of rows after merging and filtering: {len(all_filtered_jobs)}")

# Select only description column
if "Job Description" in all_filtered_jobs.columns:
    all_descriptions = all_filtered_jobs[["Job Description"]].rename(
        columns={"Job Description": "description"}
    )
elif "description" in all_filtered_jobs.columns:
    all_descriptions = all_filtered_jobs[["description"]]
elif "jobdescription" in all_filtered_jobs.columns:
    all_descriptions = all_filtered_jobs[["jobdescription"]].rename(
        columns={"jobdescription": "description"}
    )

# Remove duplicate descriptions
all_descriptions = all_descriptions.drop_duplicates()
# Drop rows with NaN in the description column
all_descriptions = all_descriptions.dropna(subset=["description"])

# Check for NaN values
print("Check NaN Values:")
print(all_descriptions.isnull().sum())

# Save the filtered job descriptions to a csv file
output_path = INTERIM_DATA_DIR / "filtered_job_descriptions.csv"
all_descriptions.to_csv(output_path, index=False)
print(f"บันทึกเฉพาะ description ลงในไฟล์: {output_path}")
