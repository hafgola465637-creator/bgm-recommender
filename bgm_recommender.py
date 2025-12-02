# bgm_recommender.py
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class BGMRecommender:
    def __init__(self, library_path="bgm_library.json", model_name="Qwen/Qwen3-Embedding-4B"):
        self.library_path = library_path
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.embeddings_dir = "embeddings"
        os.makedirs(self.embeddings_dir, exist_ok=True)
        self.load_library()

    def load_library(self):
        with open(self.library_path, "r", encoding="utf-8") as f:
            self.library = json.load(f)
        print(f"📚 已加载 {len(self.library)} 首歌曲")

    def get_embedding(self, text):
        # 自动截断或填充到模型支持长度（Qwen3 最大 32768，但我们用默认即可）
        return self.model.encode([text], normalize_embeddings=True)[0]

    def cache_all_embeddings(self):
        """为所有歌曲生成并缓存 embedding（首次运行会慢，之后很快）"""
        for i, song in enumerate(self.library):
            title = song.get("title", "未知")
            artist = song.get("artist", "未知")
            lyrics = song.get("lyrics", "")
            cache_file = os.path.join(self.embeddings_dir, f"{i}.npy")

            if not os.path.exists(cache_file):
                print(f"🔄 正在生成 [{i+1}/{len(self.library)}] {title} - {artist}")
                emb = self.get_embedding(lyrics)
                np.save(cache_file, emb)
            else:
                # 缓存已存在，跳过
                pass

    def recommend(self, query, top_k=5):
        query_emb = self.get_embedding(query)
        similarities = []

        for i, song in enumerate(self.library):
            cache_file = os.path.join(self.embeddings_dir, f"{i}.npy")
            if os.path.exists(cache_file):
                song_emb = np.load(cache_file)
                sim = cosine_similarity([query_emb], [song_emb])[0][0]
                similarities.append((sim, i))
            else:
                # 理论上不会发生，因为 cache_all_embeddings 已全覆盖
                pass

        # 按相似度排序
        similarities.sort(reverse=True, key=lambda x: x[0])
        results = []
        for sim, idx in similarities[:top_k]:
            song = self.library[idx]
            results.append({
                "title": song.get("title", "未知"),
                "artist": song.get("artist", "未知"),
                "language": song.get("language", "unknown"),
                "similarity": float(sim),
                "index": idx
            })
        return results

if __name__ == "__main__":
    recommender = BGMRecommender()
    recommender.cache_all_embeddings()  # 确保所有 embedding 已生成

    while True:
        query = input("\n🎵 请输入你想找的 BGM 关键词（输入 'quit' 退出）: ").strip()
        if query.lower() == "quit":
            break
        if not query:
            continue

        results = recommender.recommend(query, top_k=5)
        print(f"\n🔍 根据 '{query}' 推荐的 BGM：")
        for i, res in enumerate(results, 1):
            print(f"{i}. {res['title']} - {res['artist']} ({res['language']}) "
                  f"[相似度: {res['similarity']:.4f}]")