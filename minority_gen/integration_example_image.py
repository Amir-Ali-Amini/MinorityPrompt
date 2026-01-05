#!/usr/bin/env python
"""
Integration Example: MinorityPrompt Generator + Demographic Evaluator

This script shows how to use both modules together in a single workflow.
"""

import sys
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

# =============================================================================
# Quick Usage Example
# =============================================================================


def example_basic():
    """Basic usage: generate and evaluate."""
    from minority_gen import (
        generate_comparison,
        MinorityGenerator,
        ModelConfig,
        PromptOptConfig,
    )
    from minority_gen.evaluation import DemographicEvaluator
    from torchvision.utils import save_image

    # === Setup output directories with timestamp ===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"outputs/{timestamp}")
    baseline_dir = output_dir / "baseline"
    minority_dir = output_dir / "minority"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    minority_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run timestamp: {timestamp}")
    print(f"Output directory: {output_dir}")

    # === Generation ===
    prompts = [
        "Generate an image of a doctor who is smiling at the camera",
        "Generate an image of a nurse who is smiling at the camera",
    ]

    generator = MinorityGenerator(
        model_config=ModelConfig(model="sdxl_lightning"),
        popt_config=PromptOptConfig.fast(),
    )

    for i, prompt in enumerate(prompts):
        result = generator.generate(
            prompt=prompt,
            seed=42 + i,
            generate_baseline=True,
            generate_minority=True,
            modifier=
        )

        if result.baseline is not None:
            save_image(result.baseline, baseline_dir / f"sample_{i:03d}.png")

        if result.minority is not None:
            save_image(result.minority, minority_dir / f"sample_{i:03d}.png")

    print(f"\nSaved images to {output_dir}")

    # === Evaluation using saved images ===
    evaluator = DemographicEvaluator(
        face_model_path="models/shape_predictor_5_face_landmarks.dat",
        fairface_model_path="models/res34_fair_align_multi_7_20190809.pt",
    )

    result_baseline = evaluator.evaluate_directory(
        directory=baseline_dir,
        tag="baseline",
    )

    result_minority = evaluator.evaluate_directory(
        directory=minority_dir,
        tag="minority",
    )

    evaluator.print_comparison(result_baseline, result_minority)

    return result_baseline, result_minority


# =============================================================================
# Detailed Usage Example
# =============================================================================


def example_detailed(n_samples = 5):
    """
    Detailed example with custom configuration and saving results.
    """
    from torchvision.utils import save_image
    from minority_gen import (
        MinorityGenerator,
        ModelConfig,
        PromptOptConfig,
    )
    from minority_gen.prompt_modifiers import (
        CompositeModifier,
        SuffixModifier,
    )
    from minority_gen.evaluation import DemographicEvaluator

    # === Setup output directories with timestamp ===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"outputs/{timestamp}")
    baseline_dir = output_dir / "baseline"
    minority_dir = output_dir / "minority"
    modified_dir = output_dir / "modified"

    baseline_dir.mkdir(parents=True, exist_ok=True)
    minority_dir.mkdir(parents=True, exist_ok=True)
    modified_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run timestamp: {timestamp}")
    print(f"Output directory: {output_dir}")

    # === Configuration ===
    model_config = ModelConfig(
        model="sdxl_lightning",
        method="ddim_lightning",
        NFE=4,
        cfg_guidance=1.0,
    )

    popt_config = PromptOptConfig(
        enabled=True,
        p_opt_iter=10,
        p_opt_lr=0.01,
        t_lo=0.0,
        dynamic_pr=True,
        init_type="gaussian",
    )

    # === Generate Images ===
    generator = MinorityGenerator(
        model_config=model_config,
        popt_config=popt_config,
    )

    prompts = [
        "Generate an image of a doctor who is smiling at the camera",
        "Generate an image of a nurse who is smiling at the camera",
    ]

    modifier = CompositeModifier(
        [
            SuffixModifier("professional headshot"),
            SuffixModifier("studio lighting"),
        ]
    )

    print("Generating images...")
    for j, prompt in enumerate(prompts):
        print(f"  [{j+1}/{len(prompts)}] {prompt[:50]}...")

        for i in tqdm(range(n_samples)):
            result = generator.generate(
                prompt=prompt,
                modifier=modifier,
                seed=42 + i,
                generate_baseline=True,
                generate_minority=True,
            )

            if result.baseline is not None:
                save_image(result.baseline, baseline_dir / f"sample_{j:03d}_{i:03d}.png")

            if result.minority is not None:
                save_image(result.minority, minority_dir / f"sample_{j:03d}_{i:03d}.png")

            if result.modified is not None:
                save_image(result.modified, modified_dir / f"sample_{j:03d}_{i:03d}.png")

    print(f"\nSaved images to {output_dir}")

    # === Evaluate Images ===
    print("\nEvaluating images...")

    evaluator = DemographicEvaluator(
        face_model_path="models/shape_predictor_5_face_landmarks.dat",
        fairface_model_path="models/res34_fair_align_multi_7_20190809.pt",
        save_detected_faces=True,
    )

    result_baseline = evaluator.evaluate_directory(
        directory=baseline_dir,
        tag="baseline",
        output_dir=output_dir / "results",
    )

    result_minority = evaluator.evaluate_directory(
        directory=minority_dir,
        tag="minority",
        output_dir=output_dir / "results",
    )

    result_modified = evaluator.evaluate_directory(
        directory=modified_dir,
        tag="modified",
        output_dir=output_dir / "results",
    )

    # === Compare Results ===
    print("\n" + "=" * 80)
    print("BASELINE vs MINORITY")
    print("=" * 80)
    evaluator.print_comparison(result_baseline, result_minority)

    print("\n" + "=" * 80)
    print("BASELINE vs MODIFIED")
    print("=" * 80)
    evaluator.print_comparison(result_baseline, result_modified)

    # === Save Detailed Results ===
    result_baseline.save(output_dir / "results", prefix="baseline")
    result_minority.save(output_dir / "results", prefix="minority")
    result_modified.save(output_dir / "results", prefix="modified")

    print(f"\nAll results saved to: {output_dir}")

    return {
        "baseline": result_baseline,
        "minority": result_minority,
        "modified": result_modified,
    }

def sharif_task(n_samples=5, use_lightning=False):
    """
    Detailed example with custom configuration and saving results.
    """
    from torchvision.utils import save_image
    from minority_gen import (
        MinorityGenerator,
        ModelConfig,
        PromptOptConfig,
    )
    from minority_gen.prompt_modifiers import (
        CompositeModifier,
        SuffixModifier,
        SharifModifier
    )
    from minority_gen.evaluation import DemographicEvaluator

    # === Setup output directories with timestamp ===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"outputs/{timestamp}")
    baseline_dir = output_dir / "baseline"
    minority_dir = output_dir / "minority"
    modified_dir = output_dir / "modified"

    baseline_dir.mkdir(parents=True, exist_ok=True)
    minority_dir.mkdir(parents=True, exist_ok=True)
    modified_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run timestamp: {timestamp}")
    print(f"Output directory: {output_dir}")

    # === Configuration ===
    if use_lightning:
        model_config = ModelConfig(
            model="sdxl_lightning",
            method="ddim_lightning",
            NFE=4,
            cfg_guidance=1.0,
        )
    else:
        model_config = ModelConfig(
            model="sdxl",
            method="ddim",
            NFE=20,
            cfg_guidance=7.5,
        )

    popt_config = PromptOptConfig(
        enabled=True,
        p_opt_iter=20,
        p_opt_lr=0.05,
        t_lo=0.0,
        dynamic_pr=True,
        init_type="gaussian",
    )

    # === Generate Images ===
    generator = MinorityGenerator(
        model_config=model_config,
        popt_config=popt_config,
    )

    prompts = [
        "Generate an image of a doctor who is smiling at the camera",
        "Generate an image of a nurse who is smiling at the camera",
    ]

    modifier = CompositeModifier(
        [
            # SuffixModifier("professional headshot"),
            # SuffixModifier("studio lighting"),
            SharifModifier()
        ]
    )

    print("Generating images...")
    for j, prompt in enumerate(prompts):
        print(f"  [{j+1}/{len(prompts)}] {prompt[:50]}...")

        for i in tqdm(range(n_samples)):
            result = generator.generate(
                prompt=prompt,
                modifier=modifier,
                seed=42 + i,
                generate_baseline=True,
                generate_minority=True,
            )

            if result.baseline is not None:
                save_image(result.baseline, baseline_dir / f"sample_{j:03d}_{i:03d}.png")

            if result.minority is not None:
                save_image(result.minority, minority_dir / f"sample_{j:03d}_{i:03d}.png")

            if result.modified is not None:
                save_image(result.modified, modified_dir / f"sample_{j:03d}_{i:03d}.png")

    print(f"\nSaved images to {output_dir}")

    # === Evaluate Images ===
    print("\nEvaluating images...")

    evaluator = DemographicEvaluator(
        face_model_path="models/shape_predictor_5_face_landmarks.dat",
        fairface_model_path="models/res34_fair_align_multi_7_20190809.pt",
        save_detected_faces=True,
    )

    result_baseline = evaluator.evaluate_directory(
        directory=baseline_dir,
        tag="baseline",
        output_dir=output_dir / "results",
    )

    result_minority = evaluator.evaluate_directory(
        directory=minority_dir,
        tag="minority",
        output_dir=output_dir / "results",
    )

    result_modified = evaluator.evaluate_directory(
        directory=modified_dir,
        tag="modified",
        output_dir=output_dir / "results",
    )

    # === Compare Results ===
    print("\n" + "=" * 80)
    print("BASELINE vs MINORITY")
    print("=" * 80)
    evaluator.print_comparison(result_baseline, result_minority)

    print("\n" + "=" * 80)
    print("BASELINE vs MODIFIED")
    print("=" * 80)
    evaluator.print_comparison(result_baseline, result_modified)

    # === Save Detailed Results ===
    result_baseline.save(output_dir / "results", prefix="baseline")
    result_minority.save(output_dir / "results", prefix="minority")
    result_modified.save(output_dir / "results", prefix="modified")

    print(f"\nAll results saved to: {output_dir}")

    return {
        "baseline": result_baseline,
        "minority": result_minority,
        "modified": result_modified,
    }


# =============================================================================
# Metrics-Only Example (no generation)
# =============================================================================


def example_metrics_only():
    """
    Example: Just calculate metrics from existing predictions DataFrame.
    Useful if you already have demographic predictions and just want metrics.
    """
    import pandas as pd
    import numpy as np

    from minority_gen.evaluation import (
        calculate_all_metrics,
        compare_metrics,
        RACE_LABELS_7,
        GENDER_LABELS,
        AGE_LABELS,
    )

    np.random.seed(42)
    n_samples = 100

    # Biased distribution (mostly White, Male, 20-29)
    biased_df = pd.DataFrame(
        {
            "race": np.random.choice(
                RACE_LABELS_7,
                n_samples,
                p=[0.6, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05],
            ),
            "gender": np.random.choice(GENDER_LABELS, n_samples, p=[0.7, 0.3]),
            "age": np.random.choice(
                AGE_LABELS,
                n_samples,
                p=[0.02, 0.03, 0.05, 0.5, 0.2, 0.1, 0.05, 0.03, 0.02],
            ),
        }
    )

    # More uniform distribution
    uniform_df = pd.DataFrame(
        {
            "race": np.random.choice(RACE_LABELS_7, n_samples),
            "gender": np.random.choice(GENDER_LABELS, n_samples),
            "age": np.random.choice(AGE_LABELS, n_samples),
        }
    )

    metrics_biased = calculate_all_metrics(biased_df)
    metrics_uniform = calculate_all_metrics(uniform_df)

    print("Biased Distribution Metrics:")
    print(metrics_biased.summary().to_string(index=False))

    print("\nUniform Distribution Metrics:")
    print(metrics_uniform.summary().to_string(index=False))

    print("\nComparison (Biased vs Uniform):")
    comparison = compare_metrics(metrics_biased, metrics_uniform, "biased", "uniform")
    print(comparison.to_string(index=False))

    return metrics_biased, metrics_uniform


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Integration Examples")
    parser.add_argument(
        "--example",
        choices=["basic", "detailed", "metrics"],
        default="metrics",
        help="Which example to run",
    )

    args = parser.parse_args()

    if args.example == "basic":
        print("Running basic example (requires GPU and models)...")
        example_basic()
    elif args.example == "detailed":
        print("Running detailed example (requires GPU and models)...")
        example_detailed()
    elif args.example == "metrics":
        print("Running metrics-only example (no GPU required)...")
        example_metrics_only()


