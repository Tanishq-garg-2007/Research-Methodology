# Kaggle Benchmark Runner: Qwen3-VL-2B-Instruct on Small Scene Graphs

This directory contains the production-ready script to run the 100-sample Spatial Reasoning Benchmark on Kaggle GPU.

### How to Run via Kaggle CLI:
```bash
kaggle kernels push -p kaggle_runner
```

### Options:
- By default, it runs **both** Direct Prompting (~20 seconds) and Zero-Shot CoT (~60 minutes), printing the full comparative scoreboard.
- To run only Direct Prompting: Edit `eval_benchmark.py` or run `python eval_benchmark.py --strategy direct`.
