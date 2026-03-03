"""SDF-Inoculation: Fine-tune LLMs on SDF/SFT data, evaluate on emergent misalignment.

Usage:
    Training:
        python experiments/qwen32b_sdf/train.py
        python experiments/qwen32b_sorh/train.py
        python experiments/qwen32b_sdf_then_sorh/train.py --sdf-checkpoint <path>

    Evaluation:
        python experiments/qwen32b_sdf/eval.py --sampler-path <path>
        python experiments/qwen32b_sorh/eval.py --sampler-path <path>
        python experiments/qwen32b_sdf_then_sorh/eval.py --sampler-path <path>

    Comparison:
        python experiments/compare_all.py
"""
