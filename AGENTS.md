# Repository Guidelines

## Project Structure & Module Organization
`groot/` contains the Python package. Most core work happens under `groot/vla/`: `data/` for dataset loaders and transforms, `model/` for DreamZero backbones and action heads, `configs/` for Hydra and DeepSpeed config files, and `experiment/` for training entry points. Use `scripts/train/` for launch scripts, `scripts/data/` for dataset conversion utilities, and `eval_utils/` for policy server/client and sim eval helpers. Root-level `socket_test_optimized_AR.py` and `test_client_AR.py` are the current inference server and smoke-test entry points. Reference docs live in `docs/`; sample MP4 inputs live in `debug_image/`.

## Build, Test, and Development Commands
Install locally with CUDA wheels and dev tools:
```bash
pip install -e .[dev] --extra-index-url https://download.pytorch.org/whl/cu129
```
Start the WebSocket server:
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc_per_node=2 socket_test_optimized_AR.py --port 5000 --enable-dit-cache --model-path <checkpoint>
```
Run the integration smoke test against a live server:
```bash
python test_client_AR.py --host localhost --port 5000
python -m pytest test_client_AR.py -k test_ar_droid_policy_server
```
Common training entry points are `bash scripts/train/droid_training_lora.sh` and `bash scripts/train/droid_training_full_finetune_wan22.sh`.

## Coding Style & Naming Conventions
Target Python 3.11, use 4-space indentation, and format Python with `black` and `isort` before submitting:
```bash
black groot eval_utils scripts *.py
isort groot eval_utils scripts *.py
```
Follow existing naming: `snake_case` for modules, functions, YAML files, and shell scripts; `PascalCase` for classes. Keep new Hydra config names descriptive, for example `droid_relative_wan22.yaml`. Preserve the Apache/SPDX header pattern used in source files under `groot/`.

## Testing Guidelines
`pytest` is available, but coverage is currently integration-heavy rather than exhaustive. Add tests as `test_*.py` when logic can run offline; otherwise extend the existing smoke-test scripts and document required checkpoints, ports, and env vars. For training or config changes, include at least one reproducible sanity command in the PR.

## Commit & Pull Request Guidelines
Recent history uses short, imperative subjects such as `increase cache`, `clean wan5b backbone`, and `fix learning rate of lora training`. Keep commit titles concise, present tense, and scoped to one change. PRs should state the affected workflow, list changed configs or scripts, include exact reproduction commands, and note hardware assumptions for inference or training. Do not commit datasets, checkpoints, or secrets; keep local paths configurable through env vars such as `DROID_DATA_ROOT`, `OUTPUT_DIR`, `WAN_CKPT_DIR`, and `TOKENIZER_DIR`.
