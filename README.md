# BGM Recommender System

A smart BGM recommendation engine powered by **lyric semantic embedding** and **cosine similarity**, leveraging the Qwen3 large language model for state-of-the-art text representation.

## 🌟 Features

- **Semantic-Aware Recommendation**: Recommends BGM based on the semantic meaning of lyrics (not just keyword matching)
- **Efficient Embedding Caching**: Generates and caches embeddings once, enabling fast queries afterward
- **Multi-Language Support**: Works with songs in any language supported by the Qwen3 embedding model
- **Configurable Results**: Customize the number of recommended tracks
- **User-Friendly CLI**: Simple command-line interface for quick queries
- **Extensible Library**: Easy to expand the BGM library by updating the JSON file

## 📋 Prerequisites

- Python 3.8+
- Access to the Qwen3 embedding model (requires Hugging Face account, optional for model access)
- Sufficient RAM (4GB+ recommended for the 4B parameter model)

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/bgm-recommender.git
   cd bgm-recommender
   ```

2. Install required dependencies:
   ```bash
   pip install sentence-transformers numpy scikit-learn
   ```

3. (Optional) Log in to Hugging Face for model access (if required):
   ```bash
   huggingface-cli login
   ```

## 📁 BGM Library Format

The system uses a JSON file (`bgm_library.json`) to store song metadata and lyrics. Each entry must follow this structure:

```json
[
  {
    "title": "Song Title",
    "artist": "Artist Name",
    "lyrics": "Full lyrics of the song...",
    "language": "en/zh/jp/kr/etc"
  },
  {
    "title": "Another Song",
    "artist": "Another Artist",
    "lyrics": "Lyrics for the second song...",
    "language": "zh"
  }
]
```

- `title`: Song title (required)
- `artist`: Artist name (required)
- `lyrics`: Full song lyrics (required for embedding generation)
- `language`: Language tag (optional, for display purposes only)

## 💡 Usage

### 1. Prepare Your BGM Library

- Edit `bgm_library.json` to add your own songs (follow the format above)
- Or use the sample library provided (if included)

### 2. Run the Recommender

```bash
python bgm_recommender.py
```

### 3. Start Querying

- When prompted, enter keywords describing your desired BGM (e.g., "relaxing piano", "energetic workout", "sad emotional")
- Type `quit` to exit the program

### Example Workflow

```
📚 已加载 20 首歌曲
🔄 正在生成 [1/20] Clocks - Coldplay
🔄 正在生成 [2/20] River Flows in You - Yiruma
... (embedding generation completes)

🎵 请输入你想找的 BGM 关键词（输入 'quit' 退出）: relaxing piano music

🔍 根据 'relaxing piano music' 推荐的 BGM：
1. River Flows in You - Yiruma (kr) [相似度: 0.8923]
2. Canon in D - Johann Pachelbel (de) [相似度: 0.8761]
3. Luv Letter - Kashiwa Daisuke (jp) [相似度: 0.8542]
4. Weightless - Marconi Union (en) [相似度: 0.8219]
5. Moonlight Sonata - Ludwig van Beethoven (de) [相似度: 0.7985]
```

## ⚙️ Configuration

Customize the recommender by modifying the initialization parameters in `bgm_recommender.py`:

| Parameter       | Default Value          | Description                                  |
|-----------------|------------------------|----------------------------------------------|
| `library_path`  | "bgm_library.json"     | Path to your BGM library JSON file           |
| `model_name`    | "Qwen/Qwen3-Embedding-4B" | Embedding model to use (supports any SentenceTransformer-compatible model) |
| `top_k`         | 5                      | Number of recommendations to return (modify in `recommend()` call) |

Example of custom initialization:
```python
# Use a different model and library path
recommender = BGMRecommender(
    library_path="my_custom_library.json",
    model_name="all-MiniLM-L6-v2"  # Lighter, faster model
)

# Get top 10 recommendations
results = recommender.recommend(query, top_k=10)
```

## 📁 Project Structure

```
bgm-recommender/
├── bgm_recommender.py       # Main recommendation logic
├── bgm_library.json         # BGM metadata and lyrics library
├── embeddings/              # Cached embeddings (auto-generated)
│   ├── 0.npy
│   ├── 1.npy
│   └── ...
└── README.md                # Project documentation
```

## ⚠️ Notes

- **First Run Performance**: The first time you run the program, it will generate embeddings for all songs in the library. This may take several minutes (depending on the number of songs and your hardware). Subsequent runs will use cached embeddings and be much faster.
- **Model Requirements**: The default Qwen3-Embedding-4B model requires ~8GB of RAM. For resource-constrained environments, use a smaller model like `all-MiniLM-L6-v2`.
- **Lyrics Quality**: Recommendation accuracy depends on the quality and completeness of the lyrics in your library. More detailed lyrics lead to better semantic matching.

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Sentence Transformers](https://www.sbert.net/) - For easy-to-use embedding models
- [Qwen](https://github.com/QwenLM/Qwen) - For the powerful embedding model
- [Scikit-learn](https://scikit-learn.org/) - For cosine similarity calculation
- All the artists and musicians whose work makes this project meaningful

---

Made with ❤️ for music lovers and developers. Feel free to star ⭐ the repository if you find it useful!
