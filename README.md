 # PRAISE

This repository implements the **PRAISE** method for attribute-based retrieval with Test-Time Adaptation (TTA). It supports both Image-to-Image (I2I) and Text-to-Image (T2I) retrieval tasks.

---

## Table of Contents
1. [Setup](#setup)
2. [Creating Attribute Banks](#creating-attribute-banks)
3. [Testing the Model](#testing-the-model)
4. [Repository Structure](#repository-structure)

---

## Setup

To set up the environment for this repository, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd praise
   ```

2. **Create a Conda environment**:
   Use the provided `requirements.txt` to create a Conda environment named `praise`:
   ```bash
   conda create --name praise --file requirements.txt
   conda activate praise
   ```

3. **Install additional dependencies (if required)**:
   If any additional dependencies are needed, install them using `pip`:
   ```bash
   pip install -r requirements.txt
   ```

---

## Creating Attribute Banks

Before running the retrieval tasks, you need to create attribute banks for the datasets. Attribute banks are essential for the PRAISE method to function.

1. **Run the `make_attribute_bank.sh` script**:
   The script generates attribute banks for a specific dataset. For example:
   ```bash
   bash make_attribute_bank.sh
   ```

2. **Script details**:
   - The script uses the `miner.py` utility to process images and extract attributes.
   - Key arguments:
     - `--image_root`: Path to the dataset's image directory.
     - `--dataset`: Name of the dataset (e.g., `market1501`).
     - `--out_dir`: Directory to save the generated attribute bank.
     - `--sample_per_pid`: Number of samples per person ID.
     - `--max_images`: Maximum number of images to process.
     - `--device`: Device to use for processing (e.g., `cuda:0`).

   Modify the script as needed for your dataset paths and parameters.

---

## Testing the Model

Once the attribute banks are created, you can test the PRAISE method on Image-to-Image (I2I) and Text-to-Image (T2I) retrieval tasks.

1. **Run the `test_praise.sh` script**:
   The script tests the model on various datasets and configurations. For example:
   ```bash
   bash test_praise.sh
   ```

2. **Script details**:
   - The script runs the following Python files:
     - `i2i_tta_acs.py` for Image-to-Image retrieval.
     - `t2i_tta_acs.py` for Text-to-Image retrieval.
   - Key arguments:
     - `--backbone`: Backbone model to use (e.g., `plip`).
     - `--dataset`: Dataset name (e.g., `market`, `msmt`, `cuhkpedes`).
     - `--use_acs`: Whether to use Attribute Consistency Scoring (ACS).
     - `--steps`: Number of adaptation steps.
     - `--acs_alpha`: ACS alpha parameter.
     - `--acs_conf_thresh`: ACS confidence threshold.
     - `--acs_kmax`: Maximum number of ACS groups.
     - `--acs_groups`: Groups of attributes to consider.

   Modify the script as needed for your datasets and configurations.

---

## Repository Structure

Here is an overview of the repository structure:

- **TTA/**: Contains scripts for Test-Time Adaptation (TTA) for Image-to-Image retrieval.
- **TTA_t2i/**: Contains scripts for Test-Time Adaptation (TTA) for Text-to-Image retrieval.
- **backbones/**: Contains backbone models (e.g., CLIP, DINO, PLIP).
- **dataloaders/**: Contains dataset loading utilities.
- **attribute_banks/**: Stores generated attribute banks for datasets.
- **utils/**: Contains utility scripts for metrics, mining, and other tasks.

---
