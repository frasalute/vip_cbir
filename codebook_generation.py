import cv2
import numpy as np
import os
import glob
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

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
