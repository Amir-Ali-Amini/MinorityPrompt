#!/usr/bin/env python
"""
Integration Example: MinorityPrompt Generator + Demographic Evaluator

This script shows how to use both modules together in a single workflow.
"""

import sys
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from utils.maintain import clean_gpu

sys.path.insert(0, str(Path(__file__).parent))


def sharif_task(
    n_samples=5,
    use_lightning=False,
    p_opt_iter=10,
    p_opt_lr=0.01,
    t_lo=0.0,
    model="sdxl",
    seed_plus: int = 0,
    init_type="default",
    prompts=[
        "Generate an image of a doctor who is smiling at the camera",
        "Generate an image of a nurse who is smiling at the camera",
    ],
):
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
        SharifModifier,
    )
    from minority_gen.evaluation import DemographicEvaluator

    # === Setup output directories with timestamp ===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(
        f"outputs/{timestamp}_{model}{'_lightning' if use_lightning else ''}_{init_type}_{p_opt_iter=}_{p_opt_lr=}_{t_lo=}_{seed_plus=}_{n_samples=}"
    )
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
    elif model.lower() == "sdxl":
        model_config = ModelConfig(
            model="sdxl",
            method="ddim",
            NFE=50,
            cfg_guidance=7.5,
        )
    else:
        model_config = ModelConfig(
            model=model,
            method="ddim",
            NFE=50,
            cfg_guidance=7.5,
        )

    popt_config = PromptOptConfig(
        enabled=True,
        p_opt_iter=p_opt_iter,
        p_opt_lr=p_opt_lr,
        t_lo=t_lo,
        dynamic_pr=True,
        init_type=init_type,
    )

    # === Generate Images ===
    generator = MinorityGenerator(
        model_config=model_config,
        popt_config=popt_config,
    )

    modifier = CompositeModifier(
        [
            # SuffixModifier("professional headshot"),
            # SuffixModifier("studio lighting"),
            SharifModifier()
        ]
    )

    print("Generating images...")
    for j, prompt in enumerate(prompts):
        clean_gpu()
        print(f"  [{j+1}/{len(prompts)}] {prompt[:50]}...")

        for i in tqdm(range(n_samples)):
            clean_gpu()
            result = generator.generate(
                prompt=prompt,
                # modifier=modifier,
                seed=42 + i + seed_plus,
                # generate_baseline=True,
                generate_minority=True,
            )
            clean_gpu()

            if result.baseline is not None:
                save_image(
                    result.baseline, baseline_dir / f"sample_{j:03d}_{i:03d}.png"
                )

            if result.minority is not None:
                save_image(
                    result.minority, minority_dir / f"sample_{j:03d}_{i:03d}.png"
                )

            if result.modified is not None:
                save_image(
                    result.modified, modified_dir / f"sample_{j:03d}_{i:03d}.png"
                )
    clean_gpu()
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
    clean_gpu()
    # === Compare Results ===
    print("\n" + "=" * 80)
    print("BASELINE vs MINORITY")
    print("=" * 80)
    evaluator.print_comparison(result_baseline, result_minority)

    print("\n" + "=" * 80)
    print("BASELINE vs MODIFIED")
    print("=" * 80)
    evaluator.print_comparison(result_baseline, result_modified)

    print("\n" + "=" * 80)
    print("MINORITY vs MODIFIED")
    print("=" * 80)
    evaluator.print_comparison(result_minority, result_modified)

    print("\n" + "=" * 80)
    print("\n" + "=" * 80)

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
