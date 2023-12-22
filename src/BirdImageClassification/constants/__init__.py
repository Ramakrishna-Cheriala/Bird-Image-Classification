from pathlib import Path
import os

CONFIG_FILE_PATH = Path(
    r"C:\Users\ramak\OneDrive\Desktop\P2\Birds_Classification\config\config.yaml"
)
PARAMS_FILE_PATH = Path(
    r"C:\Users\ramak\OneDrive\Desktop\P2\Birds_Classification\params.yaml"
)


print(os.path.exists(CONFIG_FILE_PATH), "\n", os.path.exists(PARAMS_FILE_PATH))
