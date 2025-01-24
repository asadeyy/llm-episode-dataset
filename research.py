import numpy as np
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("cl-nagoya/sup-simcse-ja-large",device="mps")

# データファイルのパス
base_file = "base-episode.txt"
similar_file = "similarity-episode.txt"
dissimilar_file = "unsimilarity-episode.txt"


def load_and_preprocess(filepath):
    episodes = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip() 
            if not line: 
                continue 
            elements = {}
            for element in line.split(","):
                try: 
                    key, value = element.strip().split(": ", 1)
                    elements[key] = value
                except ValueError: 
                    print(f"Skipping invalid line: {line}") 
                    break 
            if elements: 
                episodes.append(elements) 

    return episodes

base_episodes = load_and_preprocess(base_file)
similar_episodes = load_and_preprocess(similar_file)
dissimilar_episodes = load_and_preprocess(dissimilar_file)

# 閾値のリスト
thresholds = np.arange(0.7, 0.91, 0.01)

# 各ファイルのエピソード数を確認
num_base = len(base_episodes)
num_similar = len(similar_episodes)
num_dissimilar = len(dissimilar_episodes)

# 最小のエピソード数に合わせてループ
num_episodes = min(num_base, num_similar, num_dissimilar)

# 結果を格納する辞書
results = {
    "accuracy": [],
    "fpr_similar": [],
    "fnr": []
}

# 各閾値で評価
for threshold in thresholds:
    tp = 0
    fp_similar = 0
    fn = 0
    for i in tqdm(range(num_episodes)):
        base = base_episodes[i]
        similar = similar_episodes[i]
        dissimilar = dissimilar_episodes[i]

        try:
          # 要素ごとの類似度計算
          element_similarities_similar = []
          element_similarities_dissimilar = []
          valid_elements = 0 # 有効な要素数をカウントする変数を追加

          for element in ["When", "Where", "Who", "What","Why","How"]:
              if base[element] == "該当なし" or similar[element] == "該当なし":
                  continue  # 「該当なし」の場合はスキップ
              if base[element] == "該当なし" or dissimilar[element] == "該当なし":
                  continue # 「該当なし」の場合はスキップ

              base_elements = [base[element] for element in ["When", "Where", "Who", "What","Why","How"] if base[element] != "該当なし"]
              similar_elements = [similar[element] for element in ["When", "Where", "Who", "What","Why","How"] if similar[element] != "該当なし"]
              dissimilar_elements = [dissimilar[element] for element in ["When", "Where", "Who", "What","Why","How"] if dissimilar[element] != "該当なし"]

              # 一度にベクトル化
              base_emb = model.encode(base_elements)
              similar_emb = model.encode(similar_elements)
              dissimilar_emb = model.encode(dissimilar_elements)
              
              valid_elements = len(base_elements) 
              element_similarities_similar = []
              element_similarities_dissimilar = []

              for j in range(valid_elements): 
                  sim_similar = util.cos_sim(base_emb[j], similar_emb[j])[0][0] 
                  sim_dissimilar = util.cos_sim(base_emb[j], dissimilar_emb[j])[0][0] 
                  element_similarities_similar.append(sim_similar)
                  element_similarities_dissimilar.append(sim_dissimilar)

              valid_elements += 1 #有効な要素数のカウントをインクリメント

          # 平均類似度の算出 (有効な要素数で割る)
          if valid_elements > 0: # ゼロ除算エラーを避けるためにチェックを追加
              avg_similarity_similar = np.mean(element_similarities_similar)
              avg_similarity_dissimilar = np.mean(element_similarities_dissimilar)
          else:
              avg_similarity_similar = 0 
              avg_similarity_dissimilar = 0 

    
          # 認証判定
          auth_result_similar = avg_similarity_similar >= threshold
          auth_result_dissimilar = avg_similarity_dissimilar >= threshold


          if auth_result_similar:
              tp += 1
          else:
              fn +=1

          if auth_result_dissimilar:
              fp_similar += 1
              
        except Exception as e: 
          print(f"Error processing episode {i+1}: {e}")  # エラーが発生したエピソード番号を表示 
          continue


    # 指標の計算
    accuracy = tp / 200
    fpr_similar = fp_similar / 200
    fnr = fn/200

    results["accuracy"].append(accuracy)
    results["fpr_similar"].append(fpr_similar)
    results["fnr"].append(fnr)


# グラフのプロット
plt.figure(figsize=(10, 6))
plt.plot(thresholds, results["accuracy"], label="認証成功率", marker="o")
plt.plot(thresholds, results["fpr_similar"], label="類似エピソード偽陽性率", marker="x")
plt.plot(thresholds, results["fnr"], label="偽陰性率", marker="^")
plt.xlabel("閾値")
plt.ylabel("割合")
plt.title("閾値に対する指標の変化")
plt.legend()
plt.grid(True)
plt.show()
