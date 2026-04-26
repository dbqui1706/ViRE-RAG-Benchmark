from rag_bench.cli import parse_args


def test_parse_args_accepts_compress_max_tokens():
    args = parse_args(["--csv", "data/CSConDa.csv", "--compress-max-tokens", "256"])

    assert args.compress_max_tokens == 256
