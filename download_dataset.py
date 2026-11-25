
from huggingface_hub import hf_hub_download
import os

REPO_ID = "opendatalab/OHR-Bench"

# Download directly to D drive
DOWNLOAD_DIR = r"D:\PDF_Analyzer_Dataset"

os.makedirs(DOWNLOAD_DIR, exist_ok=True)

print("Downloading OHR-Bench PDF collection to D drive...")

try:
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename="pdfs.zip",
        repo_type="dataset",
        local_dir=DOWNLOAD_DIR
    )

    print("PDF collection downloaded successfully!")
    print("Saved at:", path)

except Exception as error:
    print("Download failed:", error)

