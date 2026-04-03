# Contributing to ViRE

Thank you for contributing to the Vietnamese RAG Evaluation Benchmark!

## Development Setup

```bash
git clone https://github.com/dbqui1706/ViRE-RAG-Benchmark.git
cd ViRE-RAG-Benchmark
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -e ".[dev]"       # Install with dev dependencies
```

## Workflow

### Using Jules (AI Agent)

We use [Jules](https://jules.google.com) for automated tasks:

1. **Create a GitHub Issue** with the `jules` label
2. Jules will automatically create a plan and PR
3. Review the PR, request changes if needed
4. Merge when ready

**Best for:** Writing tests, adding docstrings, fixing TODOs, refactoring.

### Manual Development

1. Create a feature branch: `git checkout -b feat/your-feature`
2. Make changes following the conventions in `AGENTS.md`
3. Run tests: `pytest tests/ -v`
4. Run linting: `ruff check src/ tests/`
5. Commit & push, then open a PR

## Code Standards

- **Type hints** on all public functions
- **Docstrings** in Google style (Args, Returns, Raises)
- **Tests** for all new functionality (mock external services)
- `from __future__ import annotations` at top of every module
- Keep lines under 100 characters

## Testing

```bash
# Run full test suite
pytest tests/ -v

# Run specific test file
pytest tests/test_evaluator.py -v

# Skip slow tests
pytest tests/ -v -m "not slow"
```

Tests must be runnable **without API keys or GPU**.

## Commit Convention

```
feat: add new feature
fix: bug fix
refactor: code restructuring
test: add or update tests
docs: documentation changes
chore: maintenance tasks
```
