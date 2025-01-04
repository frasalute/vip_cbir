# Content-Based Image Retrieval (CBIR) Using Bag of Visual Words

This project implements a **Content-Based Image Retrieval (CBIR)** system based on the **Bag of Visual Words (BoW)** model. The system extracts SIFT descriptors from images, clusters them using k-means, and represents images as histograms of visual words for retrieval and evaluation.

---

## Features

- **Feature Extraction**: Extracts SIFT descriptors using OpenCV.
- **Visual Vocabulary**: Generates a codebook using k-means clustering.
- **Bag of Words Representation**: Constructs normalized histograms for each image.
- **TF-IDF Weighting**: Reweights histograms for better feature representation.
- **Similarity Measures**: Supports cosine similarity, Bhattacharyya distance, KL divergence, and common word matching.
- **Performance Metrics**: Evaluates using Mean Reciprocal Rank (MRR) and Top-3 Accuracy.

---

## Setup and Installation

1. Clone the repository and navigate to the project directory:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the CalTech 101 dataset and extract it into the project root directory as `101_ObjectCategories`.

---

## Running the Code

1. Ensure the dataset is in the `101_ObjectCategories/` directory.
2. Run the main script:
   ```bash
   python cbir_bow.py
   ```

---

## What the Script Does

1. **Feature Extraction**: Extracts SIFT descriptors from images in the dataset.
2. **Codebook Creation**: Runs k-means to create visual word clusters.
3. **Histogram Generation**: Converts descriptors into BoW histograms for each image.
4. **TF-IDF Weighting**: Optionally applies TF-IDF to the histograms.
5. **Retrieval and Evaluation**: Computes similarity scores and evaluates retrieval accuracy.

---

## Outputs

### BoW Table
A CSV file (`bow_table.csv`) containing:
- Image filenames
- Categories
- BoW histograms (normalized and TF-IDF weighted)

### Console Results
- Mean Reciprocal Rank (MRR) and Top-3 Accuracy for training and test datasets.

---

## Customization

1. **Categories**: Update the categories list in the script to include desired image classes.
2. **Number of Clusters (k)**: Modify `start_k`, `step_k`, and `end_k` in the script for optimal k-means performance.
3. **Dataset**: Replace `101_ObjectCategories` with another dataset and adjust paths and labels accordingly.

---

## Key Code Functions

### `extract_sift_descriptors(img_path, sift_detector)`
Extracts SIFT descriptors from a single image.
- **Input**: Image path
- **Output**: Descriptor matrix

### `compute_bow_histogram(descriptors, kmeans_model)`
Converts descriptors into a histogram based on the visual vocabulary.
- **Input**: Descriptor matrix, trained k-means model
- **Output**: Normalized histogram

### `evaluate_retrieval(test_table, train_table, measure)`
Evaluates retrieval accuracy using the specified similarity measure.
- **Input**: Test table, train table, similarity measure
- **Output**: MRR, Top-3 Accuracy

---

## Evaluation Metrics

- **Mean Reciprocal Rank (MRR)**: Measures the average rank position of the first correct category.
- **Top-3 Accuracy**: Percentage of queries where the correct category is in the top 3 results.

---

## Troubleshooting

1. **Empty Histograms**: Ensure images contain sufficient texture for SIFT extraction.
2. **Memory Issues**: Reduce `max_desc` or use fewer categories and images.
3. **Slow Execution**: Limit the range of `k` values in grid search or use a subset of the dataset.

---

## Requirements

This project depends on the following Python libraries:
- `opencv-contrib-python>=4.5.0` (for SIFT and other advanced OpenCV features)
- `numpy>=1.19.0` (for numerical computations)
- `pandas>=1.1.0` (for data manipulation and table creation)
- `scikit-learn>=0.24.0` (for k-means clustering and evaluation metrics)
- `matplotlib>=3.3.0` (optional, for visualization purposes)

To install these dependencies, use the provided `requirements.txt` file:
```bash
pip install -r requirements.txt
```
