#!/usr/bin/env python
"""
Example: Full Pipeline - Generation + Demographic Evaluation

This script demonstrates how to:
1. Generate images using MinorityPrompt
2. Evaluate demographic bias using the evaluation module
3. Compare baseline vs minority results

Requirements:
    - dlib: pip install dlib
    - FairFace model: res34_fair_align_multi_7_20190809.pt
    - dlib shape predictor: shape_predictor_5_face_landmarks.dat
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from typing import List, Optional


def download_models(model_dir: str = "models"):
    """Download required models if not present."""
    import urllib.request
    import bz2
    
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Download dlib shape predictor
    dlib_model = model_dir / "shape_predictor_5_face_landmarks.dat"
    if not dlib_model.exists():
        print("Downloading dlib shape predictor...")
        url = "http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2"
        bz2_path = model_dir / "shape_predictor_5_face_landmarks.dat.bz2"
        
        urllib.request.urlretrieve(url, bz2_path)
        
        # Decompress
        with bz2.open(bz2_path, 'rb') as f_in:
            with open(dlib_model, 'wb') as f_out:
                f_out.write(f_in.read())
        bz2_path.unlink()
        print(f"Saved to: {dlib_model}")
    
    # FairFace model needs to be downloaded manually from GitHub
    fairface_model = model_dir / "res34_fair_align_multi_7_20190809.pt"
    if not fairface_model.exists():
        print("\n" + "="*60)
        print("FairFace model not found!")
        print("Please download manually from:")
        print("  https://github.com/joojs/fairface")
        print(f"And save to: {fairface_model}")
        print("="*60 + "\n")
    
    return {
        'dlib': str(dlib_model) if dlib_model.exists() else None,
        'fairface': str(fairface_model) if fairface_model.exists() else None,
    }


def run_generation_evaluation():
    """Run the full generation + evaluation pipeline."""
    from minority_gen import (
        MinorityGenerator,
        ModelConfig,
        PromptOptConfig,
        GenerationResult,
    )
    from minority_gen.evaluation import (
        DemographicEvaluator,
        EvaluationResult,
    )
    from torchvision.utils import save_image
    
    # Check/download models
    models = download_models()
    if not models['dlib'] or not models['fairface']:
        print("Error: Required models not found. Please download them first.")
        return
    
    # Configuration
    prompts = [
        "A portrait of a chef in a kitchen",
        "A doctor in a hospital",
        "A teacher in a classroom",
        "A scientist in a laboratory",
        "A nurse caring for a patient",
    ]
    
    output_dir = Path("outputs/evaluation_example")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize generator
    print("\n[Step 1] Initializing generator...")
    generator = MinorityGenerator(
        model_config=ModelConfig(
            model="sdxl_lightning",
            method="ddim_lightning",
            NFE=4,
            cfg_guidance=1.0,
        ),
        popt_config=PromptOptConfig(
            enabled=True,
            p_opt_iter=5,
            t_lo=0.0,
            dynamic_pr=True,
        ),
    )
    
    # Generate images
    print("\n[Step 2] Generating images...")
    baseline_images = []
    minority_images = []
    
    baseline_dir = output_dir / "baseline"
    minority_dir = output_dir / "minority"
    baseline_dir.mkdir(exist_ok=True)
    minority_dir.mkdir(exist_ok=True)
    
    for i, prompt in enumerate(prompts):
        print(f"\n  [{i+1}/{len(prompts)}] {prompt}")
        
        # Generate baseline (no optimization)
        result = generator.generate(
            prompt=prompt,
            seed=42 + i,
            generate_baseline=True,
            generate_minority=True,
        )
        
        # Save images
        if result.baseline is not None:
            baseline_path = baseline_dir / f"sample_{i:03d}.png"
            save_image(result.baseline, baseline_path)
            baseline_images.append(str(baseline_path))
        
        if result.minority is not None:
            minority_path = minority_dir / f"sample_{i:03d}.png"
            save_image(result.minority, minority_path)
            minority_images.append(str(minority_path))
    
    print(f"\nGenerated {len(baseline_images)} baseline images")
    print(f"Generated {len(minority_images)} minority images")
    
    # Initialize evaluator
    print("\n[Step 3] Initializing evaluator...")
    evaluator = DemographicEvaluator(
        face_model_path=models['dlib'],
        fairface_model_path=models['fairface'],
        save_detected_faces=True,
    )
    
    # Evaluate baseline
    print("\n[Step 4] Evaluating baseline images...")
    result_baseline = evaluator.evaluate_images(
        images=baseline_images,
        tag="baseline",
        output_dir=output_dir / "results",
    )
    
    # Evaluate minority
    print("\n[Step 5] Evaluating minority images...")
    result_minority = evaluator.evaluate_images(
        images=minority_images,
        tag="minority",
        output_dir=output_dir / "results",
    )
    
    # Print comparison
    print("\n[Step 6] Comparing results...")
    evaluator.print_comparison(result_baseline, result_minority)
    
    # Print distributions
    print("\n\nDemographic Distributions:")
    print("\nBaseline Race Distribution:")
    if result_baseline.metrics and 'race' in result_baseline.metrics.distributions:
        print(result_baseline.metrics.distributions['race'].to_string())
    
    print("\nMinority Race Distribution:")
    if result_minority.metrics and 'race' in result_minority.metrics.distributions:
        print(result_minority.metrics.distributions['race'].to_string())
    
    print(f"\nAll results saved to: {output_dir}")


def evaluate_existing_images(
    baseline_dir: str,
    minority_dir: str,
    output_dir: str = "outputs/evaluation",
):
    """Evaluate existing image directories."""
    from minority_gen.evaluation import DemographicEvaluator
    
    models = download_models()
    if not models['dlib'] or not models['fairface']:
        print("Error: Required models not found.")
        return
    
    evaluator = DemographicEvaluator(
        face_model_path=models['dlib'],
        fairface_model_path=models['fairface'],
    )
    
    # Evaluate baseline
    print(f"\nEvaluating baseline images from: {baseline_dir}")
    result_baseline = evaluator.evaluate_directory(
        directory=baseline_dir,
        tag="baseline",
        output_dir=output_dir,
    )
    
    # Evaluate minority
    print(f"\nEvaluating minority images from: {minority_dir}")
    result_minority = evaluator.evaluate_directory(
        directory=minority_dir,
        tag="minority",
        output_dir=output_dir,
    )
    
    # Compare
    evaluator.print_comparison(result_baseline, result_minority)
    
    return result_baseline, result_minority


def evaluate_single_directory(
    image_dir: str,
    tag: str = "images",
    output_dir: str = "outputs/evaluation",
):
    """Evaluate a single directory of images."""
    from minority_gen.evaluation import DemographicEvaluator
    
    models = download_models()
    if not models['dlib'] or not models['fairface']:
        print("Error: Required models not found.")
        return
    
    evaluator = DemographicEvaluator(
        face_model_path=models['dlib'],
        fairface_model_path=models['fairface'],
    )
    
    result = evaluator.evaluate_directory(
        directory=image_dir,
        tag=tag,
        output_dir=output_dir,
    )
    
    # Print summary
    print("\n" + "="*60)
    print(f"Evaluation Summary: {tag}")
    print("="*60)
    print(f"Images processed: {result.num_images}")
    print(f"Faces detected: {result.num_faces}")
    print(f"Images with faces: {result.images_with_faces}")
    print(f"Images without faces: {result.images_without_faces}")
    
    if result.metrics:
        print("\nMetrics Summary:")
        print(result.metrics.summary().to_string(index=False))
        
        print("\nRace Distribution:")
        if 'race' in result.metrics.distributions:
            print(result.metrics.distributions['race'].to_string())
        
        print("\nGender Distribution:")
        if 'gender' in result.metrics.distributions:
            print(result.metrics.distributions['gender'].to_string())
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Demographic Evaluation for MinorityPrompt"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Full pipeline command
    gen_parser = subparsers.add_parser(
        'generate',
        help='Run full generation + evaluation pipeline'
    )
    
    # Compare directories command
    compare_parser = subparsers.add_parser(
        'compare',
        help='Compare two existing image directories'
    )
    compare_parser.add_argument('--baseline', required=True, help='Baseline images directory')
    compare_parser.add_argument('--minority', required=True, help='Minority images directory')
    compare_parser.add_argument('--output', default='outputs/evaluation', help='Output directory')
    
    # Single directory command
    eval_parser = subparsers.add_parser(
        'evaluate',
        help='Evaluate a single image directory'
    )
    eval_parser.add_argument('--images', required=True, help='Images directory')
    eval_parser.add_argument('--tag', default='images', help='Tag for this evaluation')
    eval_parser.add_argument('--output', default='outputs/evaluation', help='Output directory')
    
    # Download models command
    download_parser = subparsers.add_parser(
        'download',
        help='Download required models'
    )
    download_parser.add_argument('--dir', default='models', help='Model directory')
    
    args = parser.parse_args()
    
    if args.command == 'generate':
        run_generation_evaluation()
    elif args.command == 'compare':
        evaluate_existing_images(args.baseline, args.minority, args.output)
    elif args.command == 'evaluate':
        evaluate_single_directory(args.images, args.tag, args.output)
    elif args.command == 'download':
        download_models(args.dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
