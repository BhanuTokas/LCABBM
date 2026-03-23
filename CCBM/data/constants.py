# Modified from https://github.com/mertyg/post-hoc-cbm/blob/main
from pathlib import Path

# CUB Constants
# CUB data is downloaded from the CBM release.
# Dataset: https://worksheets.codalab.org/rest/bundles/0xd013a7ba2e88481bbc07e787f73109f5/
# Metadata and splits: https://worksheets.codalab.org/bundles/0x5b9d528d2101418b87212db92fea6683
CUB_DATA_DIR = Path("C:\\Users\\btokas\\Projects\\Datasets\\CUB_200_2011\\")
CUB_PROCESSED_DIR = CUB_DATA_DIR / "class_attr_data_10"


# Derm data constants
# Derm7pt is obtained from : https://derm.cs.sfu.ca/Welcome.html
DERM7_FOLDER = Path("C:\\Users\\btokas\\Projects\\Datasets\\derm7pt")
DERM7_META = DERM7_FOLDER / "meta" / "meta.csv"
DERM7_TRAIN_IDX = DERM7_FOLDER / "meta" / "train_indexes.csv"
DERM7_VAL_IDX = DERM7_FOLDER / "meta" / "valid_indexes.csv"

# Ham10000 can be obtained from : https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
HAM10K_DATA_DIR = "/path/to/broden/"


# BRODEN concept bank
BRODEN_CONCEPTS = Path("C:/Users/btokas/Projects/Datasets/broden_concepts/")
