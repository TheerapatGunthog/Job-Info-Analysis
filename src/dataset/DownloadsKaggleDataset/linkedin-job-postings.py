from loguru import logger
import subprocess

EXTERNAL_DATA_DIR = "../../../data/external"


def download_kaggle_dataset():
    """
    Downloads a Kaggle dataset.
    """

    dataset = "arshkon/linkedin-job-postings"
    destination = EXTERNAL_DATA_DIR + "LinkedIn Job Postings (2023 - 2024)"

    try:
        # Ensure destination directory exists
        destination.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading dataset: {dataset} to {destination}")
        # Use kaggle API to download dataset
        subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                dataset,
                "-p",
                str(destination),
                "--unzip",
            ],
            check=True,
        )
        logger.info(f"Dataset downloaded successfully to {destination}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to download dataset: {dataset}. Error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")


if __name__ == "__main__":
    download_kaggle_dataset()
