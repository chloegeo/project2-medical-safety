# Research Log

## Day 1

- Repo initialized, pushed to GitHub (private).
- Env: WSL Ubuntu + Colab Pro. No local GPU.
- Smoke tests passed:
  - Qwen/Qwen2.5-1.5B-Instruct on T4 (sanity check)
  - meta-llama/Llama-3.2-1B-Instruct on T4
  - meta-llama/Llama-3.1-8B-Instruct (4-bit) on A100
- Hook path `model.model.layers[i].mlp` works on both families.
- Observation: `captured["activation"]` has seq_len=1 after generate() because of KV caching. The real analysis script must do a single forward pass on the prompt (no generate), then pool over the last token of the prompt.