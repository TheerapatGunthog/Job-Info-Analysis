from loguru import logger
import subprocess
from pathlib import Path  # Import Path for handling directories

# Define the external data directory
EXTERNAL_DATA_DIR = Path("../../../data/external")


def download_kaggle_dataset():
    """
    Downloads a Kaggle dataset using the Kaggle CLI.
    """
    dataset = "thedevastator/jobs-dataset-from-glassdoor"
    destination = EXTERNAL_DATA_DIR / "jobs-dataset-from-glassdoor"

    try:
        # Ensure destination directory exists
        destination.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading dataset: {dataset} to {destination}")
        # Use Kaggle API to download the dataset
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
    except FileNotFoundError as e:
        logger.error(
            f"Kaggle CLI not found. Please ensure it is installed and configured. Error: {e}"
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}")


if __name__ == "__main__":
    download_kaggle_dataset()
