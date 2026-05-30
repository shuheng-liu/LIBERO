# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

**LIBERO** is a benchmark for studying knowledge transfer in multitask and lifelong robot learning. It bundles a procedural task generator built on top of [robosuite](https://github.com/ARISE-Initiative/robosuite) + MuJoCo, five curated task suites (`libero_spatial`, `libero_object`, `libero_goal`, `libero_90`, `libero_10`; `libero_100` = `libero_90` ∪ `libero_10`), three visuomotor policies (BC-RNN, BC-Transformer, BC-ViLT), and a handful of lifelong-learning algorithms (Sequential, ER, AGEM, EWC, PackNet, Multitask, SingleTask).

The library is split into two cooperating sub-packages under `libero/`:
- `libero.libero` — simulation / environment / BDDL stack (tasks, scenes, objects, robots, predicates). Implements the manipulation environments via `BDDLBaseDomain` subclasses registered in `TASK_MAPPING`.
- `libero.lifelong` — training, evaluation, datasets, policies, and lifelong-learning algorithms. Driven by Hydra configs under `libero/configs/`.

## Install & setup

```bash
conda create -n libero python=3.8.13 && conda activate libero
pip install -r requirements.txt
# torch must be installed separately because requirements.txt pins versions without CUDA index
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 \
    --extra-index-url https://download.pytorch.org/whl/cu113
pip install -e .
```

On first import of `libero.libero`, a path config is written to `~/.libero/config.yaml` (override the location with `LIBERO_CONFIG_PATH`). That file holds the `benchmark_root`, `bddl_files`, `init_states`, `datasets`, and `assets` paths used everywhere via `get_libero_path(key)`. **If a dataset/init/bddl path looks wrong, edit `~/.libero/config.yaml` rather than chasing it through the code.** Note there are *two* near-identical implementations of this config logic — `libero/libero/__init__.py` (full, with `assets`) and `libero/__init__.py`'s `configs` submodule (lighter, no `assets`); the `libero.libero` version is the one to trust.

Download datasets (prefer HuggingFace — the original mirror is unreliable):
```bash
python benchmark_scripts/download_libero_datasets.py --use-huggingface
# or --datasets libero_spatial|libero_object|libero_goal|libero_100
```

## Running training & evaluation

Training uses Hydra; CLI overrides take the form `key=value`, with grouped configs in `libero/configs/{data,policy,train,eval,lifelong}/`:

```bash
export CUDA_VISIBLE_DEVICES=0 && export MUJOCO_EGL_DEVICE_ID=0 && \
python libero/lifelong/main.py \
    seed=10000 \
    benchmark_name=LIBERO_SPATIAL \      # LIBERO_{SPATIAL,OBJECT,GOAL,10,90,100}
    policy=bc_transformer_policy \       # bc_rnn_policy | bc_transformer_policy | bc_vilt_policy
    lifelong=base                        # base | er | agem | ewc | packnet | multitask | single_task
```

Outputs land in `experiments/<BENCHMARK>/<ALGO>/<POLICY>_seed<seed>/run_<NNN>/` (the prefix grows to `experiments_finetune` / `experiments_permute<N>` / `experiments_<embedding>` depending on flags; see `create_experiment_dir` in `libero/lifelong/utils.py`).

Per-checkpoint evaluation (separate argparse CLI, not Hydra):
```bash
python libero/lifelong/evaluate.py \
    --benchmark libero_spatial --task_id 0 \
    --algo base --policy bc_transformer_policy \
    --seed 10000 --load_task 0 --device_id 0
# for --algo multitask use --ep N (in {0,5,...,50}) instead of --load_task
```

Console-script equivalents installed by `setup.py`: `lifelong.main`, `lifelong.eval`, `libero.config_copy` (copies `libero/configs/` into CWD), `libero.create_template` (scaffolds new problem classes / scenes).

There is no test suite, linter, or formatter wired up in this repo. Notebooks under `notebooks/` are the canonical "smoke tests" / walkthroughs.

## Architecture notes that span multiple files

**Registry pattern is everywhere.** Most extensible components self-register at import time via either a decorator or a metaclass. To add a new variant, importing the module is enough — but the module *must* be imported (typically from the relevant `__init__.py`).

| What | Registry | Mechanism | Registered in |
|---|---|---|---|
| Task suites (`LIBERO_SPATIAL`, …) | `BENCHMARK_MAPPING` | `@register_benchmark` | `libero/libero/benchmark/__init__.py` |
| Lifelong algos (`Sequential`, `ER`, …) | `REGISTERED_ALGOS` | `AlgoMeta` metaclass | `libero/lifelong/algos/__init__.py` |
| Policies (`BCRNNPolicy`, …) | `REGISTERED_POLICIES` | `PolicyMeta` metaclass (skips `BasePolicy`) | `libero/lifelong/models/__init__.py` |
| Problem domains (`Libero_Tabletop_Manipulation`, …) | `TASK_MAPPING` | `@register_problem` | `libero/libero/envs/problems/__init__.py` |
| Procedural scene templates | `MU_DICT` / `SCENE_DICT` | `@register_mu(scene_type=…)` | call sites in `libero/libero/benchmark/mu_creation.py` |

Lookup by name is case-insensitive everywhere (the registries lowercase the class name). `cfg.lifelong.algo`, `cfg.policy.policy_type`, and `cfg.benchmark_name` are matched against these tables.

**Lifelong training loop (`libero/lifelong/main.py`).** For each task in the benchmark sequence: build a `SequenceVLDataset` wrapping a robomimic `SequenceDataset` plus a precomputed BERT/CLIP/GPT-2/RoBERTa/one-hot task embedding (`get_task_embs` in `libero/lifelong/utils.py`); call `algo.learn_one_task(...)`, which internally runs the BC loop in `Sequential.learn_one_task` (in `libero/lifelong/algos/base.py`) and lets each algorithm hook `start_task` / `observe` / `end_task`. After every task, all seen tasks are re-evaluated to fill the forward/loss confusion matrices stored in `result_summary` and saved as `result.pt`. `Multitask` is the special case that calls `learn_all_tasks` instead.

**Policies share a common pipeline.** `BasePolicy` (in `libero/lifelong/models/base_policy.py`) owns `compute_loss` → `preprocess_input` (image augmentation, time-dim handling) → `forward` → `policy_head.loss_fn`. Each concrete policy composes `image_encoder`, `language_encoder`, `temporal_position_encoding`, and `policy_head` from sub-configs (see `libero/configs/policy/*.yaml`, which `defaults:` into the matching sub-folder). To swap in a new image encoder or head, add a YAML under the corresponding `libero/configs/policy/<group>/` and reference it.

**Environments.** `OffScreenRenderEnv` (in `libero/libero/envs/env_wrapper.py`) wraps the registered problem class chosen by `problem_name` in the BDDL file (parsed by `BDDLUtils.get_problem_info`). Evaluation spins up `env_num=20` parallel envs via `SubprocVectorEnv` (in `libero/libero/envs/venv.py`); `main.py` forces `multiprocessing.set_start_method("spawn")` because robosuite/MuJoCo are not fork-safe.

**Task ↔ BDDL ↔ init-states wiring.** Task suites in `libero_suite_task_map.py` list bare task names. The `Benchmark` class derives `bddl_file = f"{task}.bddl"`, `init_states_file = f"{task}.pruned_init"`, demo path `"{problem_folder}/{task}_demo.hdf5"`, and the language instruction by parsing the filename via `grab_language_from_filename`. Task order is permuted by `task_order_index` ∈ [0, 20] (21 fixed permutations in `task_orders`); `LIBERO_90` is locked to order 0.

**Procedural task creation.** New tasks are authored as Python: subclass `InitialSceneTemplates` (decorated with `@register_mu(scene_type=...)`), declare regions, call `register_task_info(...)`, then `generate_bddl_from_task_info(...)` to emit BDDL files. See `scripts/create_libero_task_example.py` and `notebooks/procedural_creation_walkthrough.ipynb` for the full pattern. `scripts/create_template.py` scaffolds new problem classes / scene XMLs from `templates/`.

## Things that bite

- `seq_len` vs `frame_stack` in `libero/lifelong/datasets.py:get_dataset`: robomimic's `SequenceDataset` pads them differently, so the codebase has a known issue noted in the docstring — `seq_len` is the right knob conceptually but `frame_stack` is what the code currently uses.
- `MUJOCO_EGL_DEVICE_ID` must be set alongside `CUDA_VISIBLE_DEVICES` for headless offscreen rendering on GPU; forgetting it causes silent rendering failures.
- The Hydra working directory is *not* the repo root — `main.py` calls `to_absolute_path("./bert")` (etc.) when caching HuggingFace models, so the first run writes a `bert/`/`clip/`/`gpt/` folder relative to the original CWD (these are gitignored).
- `~/.libero/config.yaml` on first import will block on `input()` asking whether to set a custom dataset path — be aware when importing in non-interactive contexts.
- `LIBERO_90` rejects any `task_order_index != 0`.
- PackNet evaluation in `evaluate.py` zeroes weights based on `previous_masks` and skips norm-layer training; replicating PackNet behavior elsewhere requires the same logic.
