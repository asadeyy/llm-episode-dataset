import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
else:
    print ("MPS device not found.")
    mps_device = torch.device("cpu")

model = SentenceTransformer("cl-nagoya/sup-simcse-ja-large", device=mps_device)

# CSVファイルの読み込み
df = pd.read_csv("dataset.csv")

# 中間結果を保存するディレクトリ
output_dir = "intermediate_results"
os.makedirs(output_dir, exist_ok=True)

# カラムごとの類似度計算関数 (SimCSEを使用)
def calculate_column_similarity(text1, text2):
    if pd.isna(text1) or pd.isna(text2) or not isinstance(text1, str) or not isinstance(text2, str) or not text1 or not text2:  # nullまたは空文字列の場合は類似度0
        return 0.0
    embeddings = model.encode([text1, text2])
    similarity = util.cos_sim(embeddings[0], embeddings[1]).item() # util.cos_simを使用し、スカラー値に変換
    return similarity

# 全体の類似度計算関数 (各カラムの平均)
def calculate_similarity(row1, row2):
    similarities = []
    for column in ['When', 'Where', 'Who', 'What', 'Why', 'How']: #　全ての対象カラム
        similarity = calculate_column_similarity(row1[column], row2[column])
        similarities.append(similarity)
    return np.mean(similarities)

# 評価実験関数 
def evaluate(df, threshold, topic):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    results = []

    first_row = df.iloc[0]  # トピックの最初の行を取得

    for i in range(1, len(df)): # 2行目から最終行までループ
        similarity = calculate_similarity(first_row, df.iloc[i])

        true_label = first_row['Label'] and df['Label'].iloc[i]
        predicted_label = similarity >= threshold

        results.append([topic, first_row['What'], df['What'].iloc[i], similarity, threshold, true_label, predicted_label])

        if predicted_label and true_label:
            tp += 1
        elif predicted_label and not true_label:
            fp += 1
        elif not predicted_label and not true_label:
            tn += 1
        elif not predicted_label and true_label:
            fn += 1

    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) != 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) != 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) != 0 else 0

    # 中間結果を保存
    intermediate_results_df = pd.DataFrame(results, columns=['Topic', 'Text1', 'Text2', 'Similarity', 'Threshold', 'True Label', 'Predicted Label'])
    intermediate_results_df.to_csv(f"{output_dir}/detailed_results_topic_{topic}_threshold_{threshold:.2f}.csv", index=False)

    return accuracy, fpr, fnr


# 閾値を変化させて評価
thresholds = np.arange(0.7, 0.91, 0.01)
results_by_topic = {}

for topic in df['Topic'].unique():
    topic_results = []
    topic_df = df[df['Topic'] == topic]

    for threshold in tqdm(thresholds, desc=f"Processing Topic {topic}"):
        accuracy, fpr, fnr = evaluate(topic_df, threshold, topic)
        topic_results.append([threshold, accuracy, fpr, fnr])

    results_by_topic[topic] = topic_results

# 全結果をデータフレームに保存
all_results = []
for topic, results in results_by_topic.items():
    for threshold, accuracy, fpr, fnr in results:
        # 中間結果を読み込み
        intermediate_df = pd.read_csv(f"{output_dir}/detailed_results_topic_{topic}_threshold_{threshold:.2f}.csv")
        all_results.extend(intermediate_df.values.tolist()) # リストに変換して追加

results_df = pd.DataFrame(all_results, columns=['Topic', 'Text1', 'Text2', 'Similarity', 'Threshold', 'True Label', 'Predicted Label'])
results_df.to_csv("detailed_results.csv", index=False)

# 指標ごとにグラフを描画
metrics = ['Accuracy', 'FPR', 'FNR']
for metric in metrics:
    plt.figure()

    for topic, results in results_by_topic.items():
        thresholds, accuracies, fprs, fnrs = zip(*results)
        if metric == 'Accuracy':
            plt.plot(thresholds, accuracies, label=f"Topic {topic}")
        elif metric == 'FPR':
            plt.plot(thresholds, fprs, label=f"Topic {topic}")
        elif metric == 'FNR':
            plt.plot(thresholds, fnrs, label=f"Topic {topic}")

    plt.xlabel("Threshold")
    plt.ylabel(metric)
    plt.title(f"{metric} vs. Threshold (by Topic)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{metric}_graph_by_topic.png")
    plt.show() 

# 指標の出力 (トピックごと)
print("\n--- 指標 (トピックごと) ---")
for topic, results in results_by_topic.items():
    print(f"\nTopic: {topic}")
    for threshold, accuracy, fpr, fnr in results:
        print(f"Threshold: {threshold:.2f}, Accuracy: {accuracy:.3f}, FPR: {fpr:.3f}, FNR: {fnr:.3f}")
