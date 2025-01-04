import cv2
import numpy as np
import pandas as pd
import os
import glob
import math
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
        descriptors = extract_sift_descriptors(img_path, sift_detector)
        histogram = compute_bow_histogram(descriptors, kmeans_model)
        data.append({
            "image_name": os.path.basename(img_path),
            "category": label, 
            "type": set_type,
            "bow_histogram": histogram
        })
    return pd.DataFrame(data)

# compute IDF vector from the training set to re-weight the histograms.
def compute_idf(train_table):
    """
    IDF(c) = log( N / (df(c) + 1) )
    df(c): number of images that have cluster c > 0
    N: total number of training images.
    """
    # Extract just the training rows
    train_only = train_table[train_table['type'] == 'train']
    bow_hists = train_only['bow_histogram'].values  # numpy arrays of shape (k,)

    num_docs = len(bow_hists)
    k = bow_hists[0].shape[0]  # number of clusters

    # Document frequency for each cluster
    df = np.zeros(k, dtype=int)

    for hist in bow_hists:
        # Count which clusters are non-zero in this histogram
        nonzero_clusters = np.where(hist > 0)[0]
        df[nonzero_clusters] += 1

    # Compute IDF
    idf = np.zeros(k, dtype=float)
    for c in range(k):
        idf[c] = math.log(num_docs / (df[c] + 1.0))

    return idf

def apply_tfidf_to_table(bow_table, idf_vector):
    """
    Apply TF-IDF weighting to each histogram in bow_table.
    TF-IDF = (raw histogram) * (IDF for each cluster).
    """
    new_hists = []
    for hist in bow_table['bow_histogram']:
        tfidf_hist = hist * idf_vector
        new_hists.append(tfidf_hist)
    bow_table['bow_histogram_tfidf'] = new_hists

    return bow_table


# Calculate similarity measures
def common_words_similarity(query_hist, db_hist):
    nonzero_query = (query_hist > 1e-12)
    nonzero_db = (db_hist > 1e-12)
    return np.sum(nonzero_query & nonzero_db)

def bhattacharyya_distance(q, p):
    """ Bhattacharyya distance = sqrt(1 - sum( sqrt(q_i * p_i) )) """
    bc = np.sum(np.sqrt(q * p))
    return np.sqrt(1.0 - bc)

def kl_divergence(q, p):
    """ Kullback-Leibler divergence = sum( q_i * log(q_i / p_i) )  """
    eps = 1e-12
    q = np.clip(q, eps, 1.0) # clip to avoid numerical issues
    p = np.clip(p, eps, 1.0)
    return np.sum(q * np.log(q / p))

# 1D array of scores (the higher the better).
# If measure is a distance (Bhattacharyya, KL), we use negative distance so that a smaller distance is equal to a larger score.
def compute_scores(
    query_hist, 
    db_hists, 
    measure="cosine"
):

    scores = []
    if measure == "cosine":
        # shape => (1, k) vs (N, k)
        sim = cosine_similarity([query_hist], db_hists)
        scores = sim[0]  # flatten
    else:
        for db_hist in db_hists:
            if measure == "common_words":
                val = common_words_similarity(query_hist, db_hist)
            elif measure == "bhattacharyya":
                dist = bhattacharyya_distance(query_hist, db_hist)
                val = -dist
            elif measure == "kl":
                dist = kl_divergence(query_hist, db_hist)
                val = -dist
            else:
                raise ValueError(f"Unknown measure: {measure}")
            
            scores.append(val)
        scores = np.array(scores)
    return scores

def evaluate_retrieval(test_table, train_table, measure):
    """
    Evaluate retrieval performance using Mean Reciprocal Rank (MRR) 
    and Top-3 Accuracy.
    """
    mrr = 0.0
    top3_count = 0
    total_queries = len(test_table)

    # Extract data from the training table
    if measure == 'cosine':
        train_hists = np.array([hist for hist in train_table['bow_histogram_tfidf']])
    else:
        train_hists = np.array([hist for hist in train_table['bow_histogram']])
    train_categories = train_table['category'].values

    for idx_query, query_row in test_table.iterrows():
        # Exclude the query image from the training database (if test_table == train_table)
        if test_table is train_table:
            mask = train_table.index != idx_query
            filtered_hists = train_hists[mask]
            filtered_categories = train_categories[mask]
        else:
            filtered_hists = train_hists
            filtered_categories = train_categories

        # Select which histogram to use for the query
        if measure == 'cosine':
            query_hist = query_row['bow_histogram_tfidf']
        else:
            query_hist = query_row['bow_histogram']

        query_category = query_row['category']

        # Compute similarity scores with all training histograms
        scores = compute_scores(query_hist, filtered_hists, measure=measure)

        # Rank indices by descending score
        ranked_indices = np.argsort(-scores)  # Negative because higher score = better rank
        ranked_categories = filtered_categories[ranked_indices]

        # Find the rank of the first correct category
        correct_positions = np.where(ranked_categories == query_category)[0]
        if len(correct_positions) > 0:
            first_correct_rank = correct_positions[0] + 1
            mrr += 1.0 / first_correct_rank

            # Check if the correct category is in the top 3
            top3_cats = ranked_categories[:3]
            if query_category in top3_cats:
                top3_count += 1
        else:
            # No correct category found
            pass

    # Compute final metrics
    mrr /= total_queries
    top3_accuracy = (top3_count / total_queries) * 100.0

    return mrr, top3_accuracy


def run_retrieval_experiment_train(bow_table, measure_list):
    train_subset = bow_table[bow_table['type'] == 'train']
    # Evaluate retrieval
    for measure in measure_list:
        mrr, top3 = evaluate_retrieval(train_subset, train_subset, measure)
        print(f"[TRAIN] Measure: {measure} => MRR={mrr:.4f}, Top-3={top3:.2f}%")

def run_retrieval_experiment_test(bow_table, measure_list):
    """
    For each measure, do retrieval where queries are the test subset,
    and the database is the train subset.
    """
    train_subset = bow_table[bow_table['type'] == 'train']
    test_subset  = bow_table[bow_table['type'] == 'test']
    for measure in measure_list:
        mrr, top3 = evaluate_retrieval(test_subset, train_subset, measure)
        print(f"[TEST]  Measure: {measure} => MRR={mrr:.4f}, Top-3={top3:.2f}%")

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

    # ---------------------------------------------------------------------
    # 6) Train KMeans with Best K and Create BoW Table
    # ---------------------------------------------------------------------
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
    # 7) TF-IDF
    # ----------------------------------------------------------------------
    idf_vector = compute_idf(bow_table)
    bow_table = apply_tfidf_to_table(bow_table, idf_vector)

    # ----------------------------------------------------------------------
    # 8) RUN THE RETRIEVAL EXPERIMENTS
    # ----------------------------------------------------------------------
    measure_list = ["common_words", "cosine", "bhattacharyya", "kl"]

    print("\n=== Retrieval on TRAIN images ===")
    run_retrieval_experiment_train(bow_table, measure_list)

    print("\n=== Retrieval on TEST images ===")
    run_retrieval_experiment_test(bow_table, measure_list)



