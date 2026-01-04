#!/usr/bin/env python
"""
Integration Example: MinorityPrompt Generator + Demographic Evaluator

This script shows how to use both modules together in a single workflow.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# =============================================================================
# Quick Usage Example
# =============================================================================

def example_basic():
    """Basic usage: generate and evaluate."""
    from minority_gen import generate_comparison, MinorityGenerator, ModelConfig, PromptOptConfig
    from minority_gen.evaluation import DemographicEvaluator, quick_evaluate
    
    # === Generation ===
    # Generate comparison images for multiple prompts
    prompts = [
        "A portrait of a chef",
        "A portrait of a doctor", 
        "A portrait of a teacher",
        "A portrait of a scientist",
    ]
    
    generator = MinorityGenerator(
        model_config=ModelConfig(model="sdxl_lightning"),
        popt_config=PromptOptConfig.fast(),
    )
    
    # Collect generated images
    baseline_tensors = []
    minority_tensors = []
    
    for i, prompt in enumerate(prompts):
        result = generator.generate(
            prompt=prompt,
            seed=42 + i,
            generate_baseline=True,
            generate_minority=True,
        )
        
        if result.baseline is not None:
            baseline_tensors.append(result.baseline)
        if result.minority is not None:
            minority_tensors.append(result.minority)
    
    # === Evaluation ===
    evaluator = DemographicEvaluator(
        face_model_path="models/shape_predictor_5_face_landmarks.dat",
        fairface_model_path="models/res34_fair_align_multi_7_20190809.pt",
    )
    
    # Evaluate tensors directly
    result_baseline = evaluator.evaluate_tensors(
        tensors=baseline_tensors,
        tag="baseline",
    )
    
    result_minority = evaluator.evaluate_tensors(
        tensors=minority_tensors,
        tag="minority",
    )
    
    # Compare and print results
    evaluator.print_comparison(result_baseline, result_minority)
    
    return result_baseline, result_minority


# =============================================================================
# Detailed Usage Example
# =============================================================================

def example_detailed():
    """
    Detailed example with custom configuration and saving results.
    """
    import torch
    from torchvision.utils import save_image
    
    from minority_gen import (
        MinorityGenerator,
        ModelConfig,
        PromptOptConfig,
    )
    from minority_gen.prompt_modifiers import (
        StyleModifier,
        CompositeModifier,
        SuffixModifier,
    )
    from minority_gen.evaluation import (
        DemographicEvaluator,
        BiasMetrics,
        calculate_all_metrics,
    )
    
    # Output directory
    output_dir = Path("outputs/detailed_evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # === Configuration ===
    model_config = ModelConfig(
        model="sdxl_lightning",
        method="ddim_lightning",
        NFE=4,
        cfg_guidance=1.0,
    )
    
    # Aggressive minority optimization
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
    
    # Test prompts focusing on people
    prompts = [
        "A portrait of a chef in a white coat",
        "A portrait of a doctor with a stethoscope",
        "A portrait of a nurse in a hospital",
        "A portrait of a teacher at a blackboard",
        "A portrait of a scientist in a lab",
        "A portrait of an engineer at work",
        "A portrait of a lawyer in an office",
        "A portrait of a business executive",
        "A portrait of an artist in a studio",
        "A portrait of a musician with an instrument",
    ]
    
    # Also test with a modifier
    modifier = CompositeModifier([
        SuffixModifier("professional headshot"),
        SuffixModifier("studio lighting"),
    ])
    
    # Generate and save images
    baseline_dir = output_dir / "images" / "baseline"
    minority_dir = output_dir / "images" / "minority"
    modified_dir = output_dir / "images" / "modified"
    
    baseline_dir.mkdir(parents=True, exist_ok=True)
    minority_dir.mkdir(parents=True, exist_ok=True)
    modified_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating images...")
    for i, prompt in enumerate(prompts):
        print(f"  [{i+1}/{len(prompts)}] {prompt[:50]}...")
        
        result = generator.generate(
            prompt=prompt,
            modifier=modifier,
            seed=42 + i,
            generate_baseline=True,
            generate_minority=True,
        )
        
        if result.baseline is not None:
            save_image(result.baseline, baseline_dir / f"sample_{i:03d}.png")
        
        if result.minority is not None:
            save_image(result.minority, minority_dir / f"sample_{i:03d}.png")
        
        if result.modified is not None:
            save_image(result.modified, modified_dir / f"sample_{i:03d}.png")
    
    # === Evaluate Images ===
    print("\nEvaluating images...")
    
    evaluator = DemographicEvaluator(
        face_model_path="models/shape_predictor_5_face_landmarks.dat",
        fairface_model_path="models/res34_fair_align_multi_7_20190809.pt",
        save_detected_faces=True,
    )
    
    # Evaluate each set
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
    print("\n" + "="*80)
    print("BASELINE vs MINORITY")
    print("="*80)
    evaluator.print_comparison(result_baseline, result_minority)
    
    print("\n" + "="*80)
    print("BASELINE vs MODIFIED")
    print("="*80)
    evaluator.print_comparison(result_baseline, result_modified)
    
    # === Save Detailed Results ===
    result_baseline.save(output_dir / "results", prefix="baseline")
    result_minority.save(output_dir / "results", prefix="minority")
    result_modified.save(output_dir / "results", prefix="modified")
    
    print(f"\nAll results saved to: {output_dir}")
    
    return {
        'baseline': result_baseline,
        'minority': result_minority,
        'modified': result_modified,
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
        calculate_bias_w,
        calculate_ens,
        calculate_kl_divergence,
        compare_metrics,
        RACE_LABELS_7,
        GENDER_LABELS,
        AGE_LABELS,
    )
    
    # Create sample data (simulating predictions)
    np.random.seed(42)
    n_samples = 100
    
    # Biased distribution (mostly White, Male, 20-29)
    biased_df = pd.DataFrame({
        'race': np.random.choice(
            RACE_LABELS_7, 
            n_samples, 
            p=[0.6, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05]  # Biased towards White
        ),
        'gender': np.random.choice(
            GENDER_LABELS,
            n_samples,
            p=[0.7, 0.3]  # Biased towards Male
        ),
        'age': np.random.choice(
            AGE_LABELS,
            n_samples,
            p=[0.02, 0.03, 0.05, 0.5, 0.2, 0.1, 0.05, 0.03, 0.02]  # Biased towards 20-29
        ),
    })
    
    # More uniform distribution
    uniform_df = pd.DataFrame({
        'race': np.random.choice(RACE_LABELS_7, n_samples),  # Uniform
        'gender': np.random.choice(GENDER_LABELS, n_samples),  # Uniform
        'age': np.random.choice(AGE_LABELS, n_samples),  # Uniform
    })
    
    # Calculate metrics
    metrics_biased = calculate_all_metrics(biased_df)
    metrics_uniform = calculate_all_metrics(uniform_df)
    
    # Print summaries
    print("Biased Distribution Metrics:")
    print(metrics_biased.summary().to_string(index=False))
    
    print("\nUniform Distribution Metrics:")
    print(metrics_uniform.summary().to_string(index=False))
    
    # Compare
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
        help="Which example to run"
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
