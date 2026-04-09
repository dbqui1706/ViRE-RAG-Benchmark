# Retrieval Benchmark Guide

Hướng dẫn chạy benchmark đánh giá các chiến lược retrieval cho Vietnamese RAG.

---

## Tổng quan

Benchmark so sánh **5 chiến lược retrieval** trên **7 bộ dữ liệu** tiếng Việt, với chunking và embedding cố định.

### Cấu hình cố định

| Tham số | Giá trị |
|---------|---------|
| Chunking | recursive (512/50) |
| Embedding | multilingual-e5-large |
| Top-K | 10 |
| Eval samples | 500/dataset (seed=42) |

### Chiến lược retrieval

| ID | Strategy | Loại | Mô tả |
|----|----------|------|-------|
| R1 | BM25 (word) | Sparse | BM25 với tokenization cấp từ (underthesea) |
| R2 | TF-IDF (word) | Sparse | TF-IDF với tokenization cấp từ (underthesea) |
| R3 | Dense | Dense | Vector similarity search (multilingual-e5-large) |
| R4 | Hybrid RRF | Hybrid | Dense + BM25 word kết hợp bằng Reciprocal Rank Fusion |
| R5 | Hybrid Weighted | Hybrid | Dense + BM25 word kết hợp bằng Weighted Sum (α=0.3) |

### Bộ dữ liệu

| Dataset | Domain |
|---------|--------|
| CSConDa | Customer Service |
| UIT-ViQuAD2 | Wikipedia QA |
| ViMedAQA_v2 | Medical QA |
| ViNewsQA | News QA |
| ViRHE4QA_v2 | Higher Education |
| ViRe4MRC_v2 | Customer Reviews |
| VlogQA_2 | Spoken QA |

---

## Chuẩn bị

### 1. Cài đặt

```bash
pip install -e ".[dev,semantic,evaluation,vietnamese]"
```

### 2. Kiểm tra

```bash
python benchmark/retrieving_benchmark.py --list
```

---

## Chạy benchmark

### Cách 1: Chạy tất cả strategies

```bash
python benchmark/retrieving_benchmark.py
```

### Cách 2: Chạy strategies cụ thể

```bash
# Chỉ chạy sparse strategies
python benchmark/retrieving_benchmark.py --strategy R1-BM25 R2-TF-IDF

# Chỉ chạy dense
python benchmark/retrieving_benchmark.py --strategy R3-Dense

# Chỉ chạy hybrid
python benchmark/retrieving_benchmark.py --strategy R4-Hybrid-RRF R5-Hybrid-Weighted

# Chạy 1 strategy bất kỳ
python benchmark/retrieving_benchmark.py --strategy R1-BM25
```

### Cách 3: Dùng bash scripts

```bash
bash bash-scripts/retrieval/run-all.sh       # Tất cả 5 strategies
bash bash-scripts/retrieval/run-sparse.sh     # R1 + R2
bash bash-scripts/retrieval/run-dense.sh      # R3
bash bash-scripts/retrieval/run-hybrid.sh     # R4 + R5
```

---

## Tuỳ chọn nâng cao

### Giới hạn eval samples

```bash
# Chạy nhanh với 100 samples/dataset
python benchmark/retrieving_benchmark.py --max-samples 100
```

### Chạy trên dataset cụ thể

```bash
python benchmark/retrieving_benchmark.py --datasets UIT-ViQuAD2 ViNewsQA
```

### Force re-run (bỏ cache)

```bash
# Không nên bỏ vì khi bỏ sẽ tạo lại vector store từ đầu nên rất lâu
python benchmark/retrieving_benchmark.py --force
```

---

## CLI Arguments

| Argument | Default | Mô tả |
|----------|---------|-------|
| `--list` | | Liệt kê strategies và datasets |
| `--strategy` | all | Strategies cần chạy (R1-BM25, R2-TF-IDF, R3-Dense, R4-Hybrid-RRF, R5-Hybrid-Weighted) |
| `--csv` | `data/processed/benchmark.csv` | Path tới benchmark CSV |
| `--datasets` | all | Filter dataset cụ thể |
| `--max-samples` | `500` | Số eval samples/dataset |
| `--output-dir` | `outputs/retrieval_benchmark` | Thư mục output |
| `--force` | `false` | Bỏ cache, rebuild tất cả |

---

## Output

Kết quả lưu vào `outputs/retrieval_benchmark/`:

```
outputs/retrieval_benchmark/
├── chroma/                          # ChromaDB index (shared, cached)
├── results/
│   ├── R1-BM25.json                 # Kết quả từng strategy
│   ├── R2-TF-IDF.json
│   ├── R3-Dense.json
│   ├── R4-Hybrid-RRF.json
│   └── R5-Hybrid-Weighted.json
├── retrieval_report.md              # Markdown report so sánh
└── summary.json                     # Tổng hợp tất cả
```

### Metrics đánh giá

| Metric | Ý nghĩa |
|--------|---------|
| Hit Rate | % câu hỏi có ít nhất 1 chunk relevant trong top-K |
| MRR | Mean Reciprocal Rank — vị trí trung bình của chunk relevant đầu tiên |
| Recall@K | % context relevant được tìm thấy trong top-K (K=1,3,5,10) |
| NDCG@K | Normalized Discounted Cumulative Gain — đánh giá cả thứ tự |

---

## Kiến trúc pipeline

```
benchmark.csv
    │
    ├── Chunking (cố định: recursive 512/50)
    │       │
    │       ▼
    │   6,134 docs → ~17K chunks
    │       │
    │       ├── Build Vectorstore (multilingual-e5-large) ──► Shared cho R3, R4, R5
    │       │
    │       ├── R1: BM25 word index
    │       ├── R2: TF-IDF word index
    │       ├── R3: Dense similarity search
    │       ├── R4: Dense + BM25 word → RRF fusion
    │       └── R5: Dense + BM25 word → Weighted Sum (α=0.3)
    │
    └── Evaluate 500 QA pairs/dataset × 7 datasets
            │
            ▼
        retrieval_report.md
```

---

## Lưu ý

- **Vectorstore shared:** Chỉ build ChromaDB index 1 lần, dùng chung cho R3, R4, R5. R1 và R2 không cần embedding.
- **Cache:** Mỗi strategy chỉ chạy 1 lần, kết quả lưu vào `results/<label>.json`. Dùng `--force` để chạy lại.
- **BM25/TF-IDF:** Đều dùng tokenization cấp từ (underthesea `word_tokenize`). Index được cache tự động.
- **Hybrid Weighted (R5):** α=0.3 nghĩa là 30% Dense + 70% BM25.
- **Hybrid RRF (R4):** Dùng `BM25WordRetriever` (không phải `BM25SylRetriever` mặc định trong `hybrid.py`).
- **GPU:** Embedding và indexing vector store cần GPU để chạy nhanh. Trên CPU sẽ rất chậm.
