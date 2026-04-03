
import pandas as pd
import pytest


@pytest.fixture
def sample_csv(tmp_path):
    """Create a small CSV matching CSConDa schema."""
    df = pd.DataFrame({
        "qid": range(1, 11),
        "question": [f"Question {i}?" for i in range(1, 11)],
        "context": [f"Context text for document {i}. " * 20 for i in range(1, 11)],
        "answer": [f"Answer {i}" for i in range(1, 11)],
    })
    path = tmp_path / "test_dataset.csv"
    df.to_csv(path, index=False)
    return path
