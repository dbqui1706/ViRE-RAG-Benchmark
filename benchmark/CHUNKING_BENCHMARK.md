# Chunking Benchmark Guide

Hướng dẫn chạy benchmark đánh giá các chiến lược chunking cho Vietnamese RAG.

---

## Tổng quan

Benchmark so sánh **5 chiến lược chunking** trên **7 bộ dữ liệu** tiếng Việt, đánh giá qua retrieval metrics (Hit Rate, MRR, NDCG@K, Recall@K).

### Chiến lược chunking

| ID | Strategy | Mô tả |
|----|----------|-------|
| C1 | `fixed` | Cắt cố định theo ký tự, không quan tâm boundary |
| C2 | `sentence` | Cắt theo câu (underthesea), bỏ qua size/overlap |
| C3 | `paragraph` | Cắt theo đoạn (double newline), bỏ qua size/overlap |
| C4 | `recursive` | Cắt đệ quy: paragraph → sentence → word |
| C5 | `semantic` | Cắt theo semantic similarity giữa các câu |

### Bộ dữ liệu

| Dataset | Domain | QA Pairs |
|---------|--------|----------|
| CSConDa | Customer Service | 1000 |
| UIT-ViQuAD2 | Wikipedia QA | 1000 |
| ViMedAQA_v2 | Medical QA | 1000 |
| ViNewsQA | News QA | 1000 |
| ViRHE4QA_v2 | Higher Education | 1000 |
| ViRe4MRC_v2 | Customer Reviews | 1000 |
| VlogQA_2 | Spoken QA | 1000 |

---

## Chuẩn bị

### 1. Cài đặt

```bash
pip install -e ".[dev,semantic,evaluation,vietnamese]"
```


### 2. Kiểm tra

```bash
python benchmark/chunking_benchmark.py --list
```

---

## Chạy benchmark

### Cách 1: Chạy experiment suite

Có **3 experiment** được định nghĩa sẵn:

| Exp | Tên | Configs | Mục đích |
|-----|-----|---------|----------|
| 1 | Chunk Size Curve | C4-256-50, C4-512-50, C4-1024-50 | So sánh kích thước chunk |
| 2 | Overlap Curve | C4-512-{0,25,50,100,200} | So sánh mức overlap |
| 3 | Method × Domain | C1-512, C2, C3, C4-512, C5 | So sánh 5 phương pháp |

```bash
# Chạy 1 experiment
python benchmark/chunking_benchmark.py --experiment 1

# Chạy nhiều experiment
python benchmark/chunking_benchmark.py --experiment 1 3

# Chạy tất cả
python benchmark/chunking_benchmark.py --experiment 1 2 3
```

### Cách 2: Chạy custom strategy

```bash
# Recursive với chunk_size=768, overlap=100
python benchmark/chunking_benchmark.py --strategy recursive --chunk-size 768 --chunk-overlap 100

# Sentence chunking
python benchmark/chunking_benchmark.py --strategy sentence

# Semantic chunking
python benchmark/chunking_benchmark.py --strategy semantic
```

### Cách 3: Dùng bash scripts

```bash
# Experiment suites
bash bash-scripts/chunking/01-chunk-size.sh
bash bash-scripts/chunking/02-overlap.sh
bash bash-scripts/chunking/03-method-domain.sh
bash bash-scripts/chunking/run-all.sh

# Individual strategies
bash bash-scripts/chunking/strategy-fixed.sh
bash bash-scripts/chunking/strategy-sentence.sh
bash bash-scripts/chunking/strategy-paragraph.sh
bash bash-scripts/chunking/strategy-recursive.sh
bash bash-scripts/chunking/strategy-semantic.sh
```

---

## Tuỳ chọn nâng cao

### Giới hạn eval samples

Mặc định evaluate 500 QA pairs/dataset:

```bash
# Chạy nhanh với 100 samples
python benchmark/chunking_benchmark.py --experiment 1 --max-samples 100
```

### Chạy trên dataset cụ thể

```bash
python benchmark/chunking_benchmark.py --experiment 3 --datasets UIT-ViQuAD2 ViNewsQA
```

### Force re-run (bỏ cache)

```bash
# Không nên bỏ vì khi bỏ sẽ lại vector store từ đầu nên rất lâu
python benchmark/chunking_benchmark.py --experiment 1 --force
```

### Custom label

```bash
python benchmark/chunking_benchmark.py --strategy recursive --chunk-size 768 --chunk-overlap 100 --label "C4-768-100-test"
```

---

## CLI Arguments

| Argument | Default | Mô tả |
|----------|---------|-------|
| `--list` | | Liệt kê strategies, experiments, datasets |
| `--experiment` | `1 2 3` | Experiment IDs (1, 2, 3) |
| `--strategy` | | Custom strategy: fixed, sentence, paragraph, recursive, semantic |
| `--chunk-size` | `512` | Kích thước chunk (ký tự) |
| `--chunk-overlap` | `50` | Overlap (ký tự) |
| `--label` | auto | Custom label cho config |
| `--csv` | `data/processed/benchmark.csv` | Path tới benchmark CSV |
| `--datasets` | all | Filter dataset cụ thể |
| `--max-samples` | `500` | Số eval samples/dataset |
| `--output-dir` | `outputs/chunking_benchmark` | Thư mục output |
| `--force` | `false` | Bỏ cache, rebuild index |

---

## Output

Kết quả lưu vào `outputs/chunking_benchmark/`:

```
outputs/chunking_benchmark/
├── chroma/                          # ChromaDB index (cached)
├── results/
│   ├── unified_C4-256-50.json       # Kết quả từng config
│   ├── unified_C4-512-50.json
│   └── ...
├── exp1_report.md                   # Markdown report
├── exp2_report.md
├── exp3_report.md
└── summary.json                     # Tổng hợp tất cả experiments
```

### Metrics đánh giá

| Metric | Ý nghĩa |
|--------|---------|
| Hit Rate | % câu hỏi có ít nhất 1 chunk relevant trong top-K |
| MRR | Mean Reciprocal Rank — vị trí trung bình của chunk relevant đầu tiên |
| Recall@K | % context relevant được tìm thấy trong top-K |
| NDCG@K | Normalized Discounted Cumulative Gain — đánh giá cả thứ tự |

---

## EDA

Phân tích dữ liệu benchmark trước khi chạy:

```bash
python benchmark/eda_benchmark.py
```

Output: `outputs/eda_benchmark/eda_report.md` + 7 biểu đồ (distribution, boxplot, histogram, chunk size coverage, v.v.)

---

## Lưu ý

- **Cache:** Mỗi config chỉ chạy 1 lần. Nếu `results/unified_<label>.json` đã tồn tại → skip. Dùng `--force` để chạy lại.
- **Reproducibility:** Eval samples được sample với `seed=42` (deterministic). Ai chạy lúc nào cũng cùng tập eval.
- **VlogQA_2:** Context rất dài (~12K chars median). Chunk size nhỏ sẽ tạo rất nhiều chunks → chậm hơn.
- **Semantic chunking (C5) và indexing vector store :** Cần GPU để chạy nhanh. Trên CPU sẽ rất chậm.
