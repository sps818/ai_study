import streamlit as st
import time
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Streamlit 标题
st.title("算法大比武 - 分类任务性能评估")


# 边栏样本数量选择
sample_sizes = st.sidebar.multiselect(
    "样本数量:",
    options=[1000, 10000, 100000, 1000000],
    default=[1000, 10000, 100000]
)

# 边栏算法选择
st.sidebar.write("算法:")
selected_algorithms = []
if st.sidebar.checkbox("Logistic Regression", value=True):
    selected_algorithms.append("Logistic Regression")
if st.sidebar.checkbox("LinearSVM", value=True):
    selected_algorithms.append("LinearSVM")
if st.sidebar.checkbox("KNN", value=True):
    selected_algorithms.append("KNN")
if st.sidebar.checkbox("Random Forest", value=True):
    selected_algorithms.append("Random Forest")

# 边栏并行运算切换按钮
parallel_execution = st.sidebar.toggle("并行运算", value=False)
n_jobs_value = -1 if parallel_execution else 1
print(n_jobs_value)


# 生成不同规模的数据集
@st.cache_data
def generate_results(sample_sizes,selected_algorithms, n_jobs_value):
    algorithms = {
        "Logistic Regression": LogisticRegression(max_iter=1000,n_jobs=n_jobs_value),
        "LinearSVM": LinearSVC(),
        "KNN": KNeighborsClassifier(n_jobs=n_jobs_value),
        "Random Forest": RandomForestClassifier(n_jobs=n_jobs_value),
    }

    results = []

    for size in sample_sizes:
        print(f"\nData Size: {size}")
        X, y = make_classification(n_samples=size, n_features=10, n_informative=8, n_redundant=2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        for name in selected_algorithms:
            model = algorithms[name]
            # 训练耗时
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time

            # 推理耗时
            start_time = time.time()
            y_pred = model.predict(X_test)
            inference_time = time.time() - start_time

            # 准确率
            accuracy = accuracy_score(y_test, y_pred)

            # 输出结果
            print(f"Algorithm: {name}")
            print(f"Training Time: {train_time:.4f} seconds")
            print(f"Inference Time: {inference_time:.4f} seconds")
            print(f"Accuracy: {accuracy:.4f}")

            # 保存结果
            results.append({
                "Data Size": size,
                "Algorithm": name,
                "Training Time (s)": train_time,
                "Inference Time (s)": inference_time,
                "Accuracy": accuracy
            })

    # 保存结果到 DataFrame
    return pd.DataFrame(results)

# 生成或获取缓存的结果
data_results = generate_results(sample_sizes,selected_algorithms, n_jobs_value)

# 显示结果表格
st.write("### Summary of Results")
st.dataframe(data_results)

# 可视化训练时间、推理时间和准确率
st.write("### Visualization of Results")

if not data_results.empty:
    '## 训练时间'
    st.bar_chart(data_results.pivot(index='Data Size', columns='Algorithm', values='Training Time (s)'),stack=False,horizontal=True)

    '## 推理时间'
    st.bar_chart(data_results.pivot(index='Data Size', columns='Algorithm', values='Inference Time (s)'),stack=False,horizontal=True)

    '## 准确率'
    st.bar_chart(data_results.pivot(index='Data Size', columns='Algorithm', values='Accuracy'),stack=False)