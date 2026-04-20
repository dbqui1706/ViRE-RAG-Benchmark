"""Generate all publication-quality figures for ViRAG-Bench paper.

Figures:
  Fig 2: Chunking strategies grouped bar chart
  Fig 3: Retrieval strategy × dataset heatmap (NDCG@5)
  Fig 5: Precision–recall tradeoff scatter (MRR vs R@10)
  Fig 6: Chunk size curve (256/512/1024)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# ── Style ────────────────────────────────────────────────────────────
FONT_SIZE = 10
DPI = 300
FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'J05__RAG_Evaluation', 'images')
os.makedirs(FIG_DIR, exist_ok=True)

matplotlib.rcParams.update({
    'font.size': FONT_SIZE,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'axes.labelsize': FONT_SIZE,
    'axes.titlesize': FONT_SIZE + 1,
    'xtick.labelsize': FONT_SIZE - 1,
    'ytick.labelsize': FONT_SIZE - 1,
    'legend.fontsize': FONT_SIZE - 2,
    'figure.dpi': DPI,
    'savefig.dpi': DPI,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.grid': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'mathtext.fontset': 'stix',
})

# Colorblind-safe palette (Tol's muted)
COLORS = ['#332288', '#88CCEE', '#44AA99', '#117733', '#999933',
          '#DDCC77', '#CC6677', '#882255', '#AA4499']


def save_fig(fig, name):
    for fmt in ['pdf', 'png']:
        path = os.path.join(FIG_DIR, f'{name}.{fmt}')
        fig.savefig(path)
    print(f'  Saved: {name}.pdf / .png')


# ═══════════════════════════════════════════════════════════════════════
# Fig 2: Chunking Strategies — Grouped Bar Chart
# ═══════════════════════════════════════════════════════════════════════
def fig2_chunking_bar():
    print('[Fig 2] Chunking strategies bar chart...')
    strategies = ['Fixed-Size', 'Sentence', 'Paragraph',
                  'Recursive', 'Semantic']
    # Best config per strategy from chunking_full_metrics_report.md
    mrr   = [47.42, 52.43, 51.57, 44.83, 50.34]
    ndcg5 = [35.80, 30.21, 52.97, 39.28, 40.53]
    r10   = [37.72, 21.15, 64.68, 44.11, 47.29]
    map5  = [30.68, 22.58, 50.64, 35.79, 35.37]

    x = np.arange(len(strategies))
    width = 0.19
    fig, ax = plt.subplots(1, 1, figsize=(7, 3.5))

    bars1 = ax.bar(x - 1.5*width, mrr,   width, label='MRR',      color=COLORS[0])
    bars2 = ax.bar(x - 0.5*width, ndcg5, width, label='NDCG@5',   color=COLORS[1])
    bars3 = ax.bar(x + 0.5*width, r10,   width, label='R@10',     color=COLORS[2])
    bars4 = ax.bar(x + 1.5*width, map5,  width, label='MAP@5',    color=COLORS[5])

    ax.set_ylabel('Score')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.legend(frameon=False, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.15))
    ax.set_ylim(0, 72)

    # Highlight best per metric
    for bars, vals in [(bars1, mrr), (bars2, ndcg5), (bars3, r10), (bars4, map5)]:
        best_idx = np.argmax(vals)
        bar = bars[best_idx]
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{vals[best_idx]:.2f}', ha='center', va='bottom',
                fontsize=FONT_SIZE - 2, fontweight='bold')

    fig.tight_layout()
    save_fig(fig, 'fig2_chunking_bar')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Fig 3: Retrieval Heatmap — NDCG@5 (strategy × dataset)
# ═══════════════════════════════════════════════════════════════════════
def fig3_retrieval_heatmap():
    print('[Fig 3] Retrieval heatmap...')
    strategies = [
        'Hybrid-W (e5)',
        'Hybrid-RRF (e5)',
        'Hybrid-W (bi)',
        'BM25',
        'Dense (e5)',
        'Hybrid-RRF (bi)',
        'TF-IDF',
        'Dense (bi)',
    ]
    datasets = ['CSConDa', 'ViQuAD2', 'ViMedAQA', 'ViNewsQA', 'ViRHE4QA', 'ViRe4MRC', 'VlogQA']

    # NDCG@5 from retrieval_full_metrics_report.md
    data = np.array([
        [26.60, 89.69, 83.10, 74.99, 81.81, 12.27, 42.27],  # Hybrid-W e5
        [24.31, 89.93, 88.01, 71.10, 78.25, 16.15, 38.30],  # Hybrid-RRF e5
        [25.50, 88.03, 78.98, 72.82, 77.98, 11.21, 38.48],  # Hybrid-W bi
        [23.88, 82.45, 72.90, 70.86, 76.62,  9.09, 35.24],  # BM25
        [17.50, 88.22, 88.89, 59.43, 69.18, 17.36, 27.70],  # Dense e5
        [20.78, 84.13, 79.30, 65.74, 69.84, 12.30, 31.53],  # Hybrid-RRF bi
        [20.32, 67.97, 68.33, 48.06, 61.30,  7.39, 16.69],  # TF-IDF
        [11.78, 72.04, 72.08, 47.24, 54.56, 11.11, 15.72],  # Dense bi
    ])

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    im = ax.imshow(data, cmap='YlOrRd', aspect='auto', vmin=5, vmax=95)

    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, rotation=35, ha='right')
    ax.set_yticks(range(len(strategies)))
    ax.set_yticklabels(strategies)

    # Annotate cells
    for i in range(len(strategies)):
        for j in range(len(datasets)):
            val = data[i, j]
            color = 'white' if val > 65 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=FONT_SIZE - 2, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('NDCG@5')

    fig.tight_layout()
    save_fig(fig, 'fig3_retrieval_heatmap')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Fig 5: Precision–Recall Scatter (MRR vs R@10)
# ═══════════════════════════════════════════════════════════════════════
def fig5_precision_recall_scatter():
    print('[Fig 5] Precision-recall scatter...')
    # Best config per strategy
    labels = ['Fixed-Size', 'Sentence', 'Paragraph',
              'Recursive-256', 'Recursive-512', 'Recursive-1024', 'Semantic']
    mrr  = [47.42, 52.43, 51.57, 53.63, 47.97, 44.83, 50.34]
    r10  = [37.72, 21.15, 64.68, 29.60, 37.29, 44.11, 47.29]
    markers = ['s', '^', 'D', 'o', 'o', 'o', 'p']
    colors  = [COLORS[0], COLORS[1], COLORS[2], COLORS[3], COLORS[4], COLORS[5], COLORS[6]]

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4))
    for i, (lbl, m, r) in enumerate(zip(labels, mrr, r10)):
        ax.scatter(r, m, c=colors[i], marker=markers[i], s=100, zorder=5,
                   edgecolors='black', linewidths=0.5)
        offset_x, offset_y = 1.5, 0.8
        if 'Sentence' in lbl:
            offset_x = -10
            offset_y = -2
        elif 'Paragraph' in lbl:
            offset_x = -4
            offset_y = 1.5
        ax.annotate(lbl, (r, m), textcoords="offset points",
                    xytext=(offset_x, offset_y), fontsize=FONT_SIZE - 2)

    ax.set_xlabel('Recall@10')
    ax.set_ylabel('MRR')
    ax.set_xlim(15, 70)
    ax.set_ylim(40, 56)

    # Add quadrant lines
    ax.axhline(y=49, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
    ax.axvline(x=40, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)

    fig.tight_layout()
    save_fig(fig, 'fig5_precision_recall')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Fig 6: Chunk Size Curve (Recursive, overlap=50)
# ═══════════════════════════════════════════════════════════════════════
def fig6_chunk_size_curve():
    print('[Fig 6] Chunk size curve...')
    sizes = [256, 512, 1024]
    mrr   = [53.63, 47.97, 44.83]
    ndcg5 = [34.93, 36.25, 39.28]
    r10   = [29.60, 37.29, 44.11]

    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
    ax.plot(sizes, mrr,   'o-', color=COLORS[0], label='MRR',    linewidth=2, markersize=7)
    ax.plot(sizes, ndcg5, 's-', color=COLORS[1], label='NDCG@5', linewidth=2, markersize=7)
    ax.plot(sizes, r10,   'D-', color=COLORS[2], label='R@10',   linewidth=2, markersize=7)

    ax.set_xlabel('Chunk Size (characters)')
    ax.set_ylabel('Score')
    ax.set_xticks(sizes)
    ax.legend(frameon=False, loc='center right')

    # Annotate endpoints
    for vals, offset in [(mrr, (5, 3)), (ndcg5, (5, -8)), (r10, (5, 3))]:
        for i in [0, 2]:  # first and last point
            ax.annotate(f'{vals[i]:.2f}', (sizes[i], vals[i]),
                        textcoords="offset points", xytext=offset,
                        fontsize=FONT_SIZE - 2)

    fig.tight_layout()
    save_fig(fig, 'fig6_chunk_size_curve')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print(f'Output: {os.path.abspath(FIG_DIR)}\n')
    fig2_chunking_bar()
    fig3_retrieval_heatmap()
    fig5_precision_recall_scatter()
    fig6_chunk_size_curve()
    print('\nAll figures generated.')
