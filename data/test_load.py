import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. 載入資料
path = r"D:\College\4th grade\機器學習在教育上的應用\student-ml-project\data\student-mat.csv"
df = pd.read_csv(path, sep=";")

# --- [重點 1：資料探索視覺化 (EDA)] ---
# 這張圖放在 PPT 第六頁
plt.figure(figsize=(8, 5))
sns.histplot(df['G3'], bins=20, kde=True, color='skyblue')
plt.title("Distribution of Final Grades (G3)")
plt.xlabel("Final Grade (G3)")
plt.ylabel("Number of Students")
plt.axvline(x=10, color='red', linestyle='--', label='Pass/Fail Threshold')
plt.legend()
plt.show()

# --- [重點 2：資料前處理] ---
df['pass'] = (df['G3'] >= 10).astype(int)
features = ['studytime', 'failures', 'absences', 'G1', 'G2']
X = df[features]
y = df['pass']

# --- [重點 3：監督式學習 - 預測及格與否] ---
print("\n--- 監督式學習：Logistic Regression ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("準確率 (Accuracy):", accuracy_score(y_test, y_pred))

# 截圖 1：混淆矩陣 (放在 PPT 第八頁上方)
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Student Pass Prediction")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 截圖 2：特徵重要度 (放在 PPT 第八頁下方或說明文字旁)
# 這張圖能證明你說的「G2 最重要」、「Failures 是負向指標」
coefficients = pd.DataFrame({'Feature': features, 'Importance': model.coef_[0]})
coefficients = coefficients.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(8, 5))
sns.barplot(x='Importance', y='Feature', data=coefficients, palette='magma')
plt.title("Feature Importance (Logistic Regression Coefficients)")
plt.show()

# --- [重點 4：非監督式學習 - 學生分群分析] ---
print("\n--- 非監督式學習：K-means Clustering ---")
cluster_features = df[['absences', 'G3']]
scaler = StandardScaler()
cluster_scaled = scaler.fit_transform(cluster_features)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(cluster_scaled)

# 截圖 3：分群散佈圖 (放在 PPT 第十頁)
plt.figure(figsize=(8, 6))
plt.scatter(df['absences'], df['G3'], c=df['cluster'], cmap='viridis', alpha=0.6)
plt.title("Student Clustering (Absences vs G3)")
plt.xlabel("Absences")
plt.ylabel("Final Grade (G3)")
plt.colorbar(label='Cluster ID')
plt.show()