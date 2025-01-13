# Plagiarism Detection with Proximity Measures and BERT

This project detects plagiarism in academic abstracts using proximity measures and a fine-tuned BERT model.

## Features
- **Proximity Measures**: Cosine Similarity, Euclidean Distance, Correlation Coefficient
- **BERT Classification**: Fine-tuned BERT for improved accuracy
- **Data Augmentation**: EDA and Parrot paraphraser
- **Threshold Optimization**: ROC-based threshold selection for proximity methods
- **Web Crawling**: Automated data collection from Google Scholar

---

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Download the BERT model:
    python -m transformers.cli download bert-base-uncased


## Usage
Data Crawling
   ```bash
    python crawling_selenium.py
   ```

Train the BERT Model
   ```bash
    python PDBert.py --mode train --data_dir ./data --output_dir ./output

   ```

Threshold Optimization
   ```bash
    python get_threshold.py
   ```
