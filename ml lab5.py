
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

# ========== CONFIGURATION ==========
CSV_PATH = "sample_0_venom_nonvenom.csv"  # path to your CSV file
IMAGE_DIR = "C:\Users\year3\Desktop\snake\Images"  # folder where all images are copied

# ========== STEP 1: Load Data ==========
df = pd.read_csv(CSV_PATH)
features = []
labels = []

print("Extracting features using ResNet50...")
model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

for idx, row in tqdm(df.iterrows(), total=len(df)):
    img_path = os.path.join(IMAGE_DIR, row["filename"])
    if not os.path.exists(img_path):
        continue
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feat = model.predict(x)[0]
        features.append(feat)
        labels.append(row["poisonous"])
    except Exception as e:
        print(f"Error loading {img_path}: {e}")

X = np.array(features)
y = np.array(labels)

# ========== A1: Logistic Regression with One Feature ==========
print("\nA1: Logistic Regression with One Feature")
X_train, X_test, y_train, y_test = train_test_split(X[:, [0]], y, test_size=0.2, random_state=42)
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)

print("Train Accuracy:", accuracy_score(y_train, y_pred_train))
print("Test Accuracy:", accuracy_score(y_test, y_pred_test))

# ========== A2: Metrics ==========
def print_metrics(y_true, y_pred, label):
    print(f"\n{label} Metrics:")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

print_metrics(y_train, y_pred_train, "Train")
print_metrics(y_test, y_pred_test, "Test")

# ========== A3: Logistic Regression with All Features ==========
print("\nA3: Logistic Regression with All Features")
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X, y, test_size=0.2, random_state=42)
clf_all = LogisticRegression(max_iter=1000)
clf_all.fit(X_train_full, y_train_full)
y_pred_train_full = clf_all.predict(X_train_full)
y_pred_test_full = clf_all.predict(X_test_full)

print_metrics(y_train_full, y_pred_train_full, "Train (All Features)")
print_metrics(y_test_full, y_pred_test_full, "Test (All Features)")
# ========== A4: KMeans Clustering ==========
print("\nA4: KMeans Clustering (k=2)")
kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto")
kmeans.fit(X_train_full)
print("Cluster Centers:", kmeans.cluster_centers_)

# ========== A5: Clustering Metrics ==========
print("\nA5: Clustering Evaluation")
labels_kmeans = kmeans.labels_
print("Silhouette Score:", silhouette_score(X_train_full, labels_kmeans))
print("Calinski-Harabasz Score:", calinski_harabasz_score(X_train_full, labels_kmeans))
print("Davies-Bouldin Index:", davies_bouldin_score(X_train_full, labels_kmeans))

# ========== A6: Clustering for Different k ==========
print("\nA6: Clustering with Multiple k-values")
sil_scores, ch_scores, db_scores = [], [], []
k_range = range(2, 10)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X_train_full)
    sil_scores.append(silhouette_score(X_train_full, km.labels_))
    ch_scores.append(calinski_harabasz_score(X_train_full, km.labels_))
    db_scores.append(davies_bouldin_score(X_train_full, km.labels_))

plt.figure()
plt.plot(k_range, sil_scores, label='Silhouette')
plt.plot(k_range, ch_scores, label='CH Score')
plt.plot(k_range, db_scores, label='DB Index')
plt.xlabel("k")
plt.ylabel("Score")
plt.title("Clustering Evaluation Metrics vs k")
plt.legend()
plt.show()

# ========== A7: Elbow Plot ==========
print("\nA7: Elbow Plot")
inertias = []
for k in range(2, 20):
    km = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X_train_full)
    inertias.append(km.inertia_)

plt.figure()
plt.plot(range(2, 20), inertias, marker='o')
plt.title('Elbow Method: Inertia vs k')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.show()



