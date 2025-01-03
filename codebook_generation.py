import cv2
import numpy as np
import pandas as pd
import os
import glob
import ast
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

def extract_sift_descriptors(img_path, sift_detector):
    """Extract SIFT descriptors from a single image."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    keypoints, descriptors = sift_detector.detectAndCompute(img, None)
    return descriptors

def compute_bow_histogram(descriptors, kmeans_model):
    """Convert a set of descriptors into a BoW histogram using a trained KMeans model."""
    if descriptors is None or descriptors.shape[0] == 0:
        # Handle case of no descriptors
        return np.zeros(kmeans_model.n_clusters, dtype=float)
    
    cluster_ids = kmeans_model.predict(descriptors)
    bow_hist = np.bincount(cluster_ids, minlength=kmeans_model.n_clusters).astype(float)
    
    # Normalize histogram 
    bow_hist /= (bow_hist.sum() + 1e-7)
    return bow_hist

def create_table (image_paths, labels, set_type, kmeans_model, sift_detector):
    """ Create a table with BoW histograms for all images"""
    data = []

    for img_path, label in zip(image_paths, labels):
        description = extract_sift_descriptors(img_path, sift_detector)
        histogram = compute_bow_histogram(desc, kmeans_model)
        data.append({
            "image_name": os.path.basename(img_path),
            "category": label, 
            "type": set_type,
            "bow_histogram": histogram
        })
    return pd.DataFrame(data)

if __name__ == "__main__":
    # ---------------------------------------------------------------------
    # 1) Gather Image Paths and Labels
    # ---------------------------------------------------------------------
    data_dir = '101_ObjectCategories'
    categories = ['airplanes', 'anchor', 'butterfly', 'panda', 'wild_cat', 'starfish', 'ant', 'barrel', 'beaver', 'brain', 'faces', 'ferry', 'helicopter', 'laptop', 'llama', 'snoopy', 'chair', 'crab', 'elephant', 'wrench']
    image_paths = []
    labels = []
    
    for cat in categories:
        cat_path = os.path.join(data_dir, cat)
        img_files = glob.glob(os.path.join(cat_path, '*.jpg'))
        for f in img_files:
            image_paths.append(f)
            labels.append(cat)

    image_paths = np.array(image_paths)
    labels = np.array(labels)

# Check how many images and labels were found
print("Number of image paths:", len(image_paths))
print("Categories:", categories)

# split into training and validation set
train_set, val_set, train_labels, val_labels = train_test_split(
    image_paths, labels, test_size=0.2, random_state=42, stratify=labels
)

print("Train size:", len(train_set), "Val size:", len(val_set))

# ---------------------------------------------------------------------
# 2) Extract ALL Descriptors from the Training Set
# ---------------------------------------------------------------------
sift = cv2.SIFT_create()
all_train_desc = []

for path in train_set:
    desc = extract_sift_descriptors(path, sift)
    if desc is not None:
        all_train_desc.append(desc)

# Combine descriptors
all_train_desc = np.vstack(all_train_desc)
print("Total descriptors shape (training):", all_train_desc.shape)
max_desc = 100000
if all_train_desc.shape[0] > max_desc:
    idx = np.random.choice(all_train_desc.shape[0], max_desc, replace=False)
    all_train_desc = all_train_desc[idx]
"""
# ---------------------------------------------------------------------
# 3) Define Grid for k
# ---------------------------------------------------------------------
start_k = 500   
step_k = 50     
end_k = 2000   

best_k = None
best_accuracy = 0.0
k_to_accuracy = {}

# ---------------------------------------------------------------------
# 4) Grid Search Over k
# ---------------------------------------------------------------------
for k in range(start_k, end_k, step_k):
    print(f"\n[GRID SEARCH] Trying k = {k} ...")

    # -- 4.1) Train KMeans on the sampled descriptors
    kmeans = KMeans(n_clusters=k, random_state=42, verbose=0, max_iter=500)
    kmeans.fit(all_train_desc)
    print(f"K={k}, inertia={kmeans.inertia_}")
        
    # -- 4.2) Build BoW histograms for train set
    train_bow = []
    for path in train_set:
        desc = extract_sift_descriptors(path, sift)
        hist = compute_bow_histogram(desc, kmeans)
        train_bow.append(hist)
    train_bow = np.array(train_bow)

    # -- 4.3) Build BoW histograms for validation set
    val_bow = []
    for path in val_set:
        desc = extract_sift_descriptors(path, sift)
        hist = compute_bow_histogram(desc, kmeans)
        val_bow.append(hist)
    val_bow = np.array(val_bow)

    # -- 4.4) Train Classifier and then evaluate
    # Use Logistic Regression
    classifier = SVC(kernel='rbf',C=1.5, gamma='scale', random_state=42)
    classifier.fit(train_bow, train_labels)
    val_preds = classifier.predict(val_bow)
    accuracy = accuracy_score(val_labels, val_preds)

    k_to_accuracy[k] = accuracy

    print(f"Accuracy for k = {k}: {accuracy:.4f}")

    # -- 4.5) Track the best k so far
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k

# ---------------------------------------------------------------------
# 5) Report Best k
# ---------------------------------------------------------------------
print("\nGrid search complete.")
for k, acc in k_to_accuracy.items():
    print(f"k = {k} => accuracy = {acc:.4f}")
print(f"\nBest k = {best_k} with accuracy = {best_accuracy:.4f}")

# ---------------------------------------------------------------------
# Results 
"""
"""Grid search complete.
k = 500 => accuracy = 0.6511
k = 550 => accuracy = 0.6532
k = 600 => accuracy = 0.6511
k = 650 => accuracy = 0.6574
k = 700 => accuracy = 0.6681
k = 750 => accuracy = 0.6596
k = 800 => accuracy = 0.6468
k = 850 => accuracy = 0.6596
k = 900 => accuracy = 0.6617
k = 950 => accuracy = 0.6489
k = 1000 => accuracy = 0.6638
k = 1050 => accuracy = 0.6532
k = 1100 => accuracy = 0.6574
k = 1150 => accuracy = 0.6638
k = 1200 => accuracy = 0.6574
k = 1250 => accuracy = 0.6660
k = 1300 => accuracy = 0.6617
k = 1350 => accuracy = 0.6468
k = 1400 => accuracy = 0.6553
k = 1450 => accuracy = 0.6617
k = 1500 => accuracy = 0.6489
k = 1550 => accuracy = 0.6404
k = 1600 => accuracy = 0.6511
k = 1650 => accuracy = 0.6404
k = 1700 => accuracy = 0.6532
k = 1750 => accuracy = 0.6404
k = 1800 => accuracy = 0.6489
k = 1850 => accuracy = 0.6468
k = 1900 => accuracy = 0.6404
k = 1950 => accuracy = 0.6553

Best k = 700 with accuracy = 0.6681"""



# ---------------------------------------------------------------------
# 6) Train KMeans with Best K and Create BoW Table
# ---------------------------------------------------------------------
best_k = 700
print(f"\n[Training Final KMeans] Using best k = {best_k} ...")
final_kmeans = KMeans(n_clusters=best_k, random_state=42, max_iter=500)
final_kmeans.fit(all_train_desc)

print("\n[CREATING BoW TABLE]")
train_table = create_table(train_set, train_labels, "train", final_kmeans, sift)
val_table = create_table(val_set, val_labels, "test", final_kmeans, sift)

bow_table = pd.concat([train_table, val_table]) # combine train and validation tables

# Save to CVS so can be consulted 
output_file = "bow_table.csv"
bow_table.to_csv(output_file, index=False)
print(f"BoW table saved to {output_file}")

# ----------------------------------------------------------------------
# 7) Mean Reciprocal Rank and Top 3 Accuracy
# ----------------------------------------------------------------------

def compute_similarity(query_hist, dataset_hists):
    """Compute similarity between a query histogram and dataset histograms."""
    similarities = cosine_similarity([query_hist], dataset_hists)
    return similarities[0]  # Flatten to 1D array

def evaluate_retrieval(test_table, train_table):
    """Evaluate retrieval performance using MRR and Top-3 Accuracy."""
    mrr = 0.0
    top3_count = 0
    total_queries = len(test_table)

    # Use `bow_histogram` column directly as it already contains numeric arrays
    train_hists = np.array([hist for hist in train_table['bow_histogram']])
    train_categories = train_table['category'].values

    for _, query_row in test_table.iterrows():
        query_hist = query_row['bow_histogram']  # Directly use the histogram
        query_category = query_row['category']

        # Compute similarities to all training images
        similarities = compute_similarity(query_hist, train_hists)

        # Rank indices by similarity (higher is better for cosine similarity)
        ranked_indices = np.argsort(-similarities)
        ranked_categories = train_categories[ranked_indices]

        # Find the rank of the first correct category
        try:
            rank = np.where(ranked_categories == query_category)[0][0] + 1  # Add 1 for 1-based rank
            mrr += 1 / rank

            # Check if the correct category is in the top 3
            if query_category in ranked_categories[:3]:
                top3_count += 1
        except IndexError:
            raise ValueError(f"No correct category found for query '{query_category}' in the ranked results.")

    # Compute final metrics
    mrr /= total_queries
    top3_accuracy = (top3_count / total_queries) * 100

    return mrr, top3_accuracy

