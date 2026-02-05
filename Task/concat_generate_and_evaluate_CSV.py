#!/usr/bin/env python
"""
Image Generation and Evaluation Pipeline
Usage:
    # Edit DEFAULT_CONFIG below, then run:
    python concat_generate_and_evaluate.py
    # Or pass args:
    python concat_generate_and_evaluate.py --original-csv prompts.csv --n-samples 10
"""
import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).parent))
from minority_gen import MinorityGenerator, ModelConfig, PromptOptConfig
from minority_gen.prompt_modifiers import CSVModifier
from minority_gen.evaluation import DemographicEvaluator

# =============================================================================
# DEFAULT CONFIG - Edit for Colab (no args needed)
# =============================================================================
DEFAULT_CONFIG = {
    "original_csv": "sample_original_prompts.csv",
    "prompt_col": "prompt",
    "enhanced_csv": None,
    "enhanced_col": "enhanced_prompt",
    "model": "sdxl",
    "use_lightning": False,
    "n_samples": 1,
    "seed_start": 42,
    "p_opt_iter": 10,
    "p_opt_lr": 0.01,
    "t_lo": 0.0,
    "init_type": "default",
    "metrics_base_dir": "./metrics_results",
    "output_dir": "./outputs/csv/main",
    "experiment_name": None,
    "start_from_row": 0,
    "face_model_path": "models/shape_predictor_5_face_landmarks.dat",
    "fairface_model_path": "models/res34_fair_align_multi_7_20190809.pt",
}


def parse_args():
    parser = argparse.ArgumentParser()
    d = DEFAULT_CONFIG
    parser.add_argument("--original-csv", default=d["original_csv"])
    parser.add_argument("--prompt-col", default=d["prompt_col"])
    parser.add_argument("--enhanced-csv", default=d["enhanced_csv"])
    parser.add_argument("--enhanced-col", default=d["enhanced_col"])
    parser.add_argument("--model", default=d["model"])
    parser.add_argument(
        "--use-lightning", action="store_true", default=d["use_lightning"]
    )
    parser.add_argument("--face_model_path", type=str, default=d["face_model_path"])
    parser.add_argument(
        "--fairface_model_path", type=str, default=d["fairface_model_path"]
    )
    parser.add_argument("--n-samples", "-n", type=int, default=d["n_samples"])
    parser.add_argument("--seed-start", type=int, default=d["seed_start"])
    parser.add_argument("--p-opt-iter", type=int, default=d["p_opt_iter"])
    parser.add_argument("--p-opt-lr", type=float, default=d["p_opt_lr"])
    parser.add_argument("--t-lo", type=float, default=d["t_lo"])
    parser.add_argument("--init-type", default=d["init_type"])
    parser.add_argument("--metrics-base-dir", default=d["metrics_base_dir"])
    parser.add_argument("--output-dir", default=d["output_dir"])
    parser.add_argument("--experiment-name", default=d["experiment_name"])
    parser.add_argument("--start-from-row", type=int, default=d["start_from_row"])
    return parser.parse_args()


def evaluate_single_image(evaluator, img_path, output_dir=None):
    """Evaluate a single image and return metrics dict."""
    if img_path is None or not Path(img_path).exists():
        return {}

    try:
        res = evaluator.evaluate_images(
            images=[str(img_path)],
            tag=Path(img_path).stem,
            output_dir=output_dir,
            progress=False,
        )

        metrics = {
            "n_faces": res.num_faces,
            "has_face": res.num_faces > 0,
        }

        # Get per-face predictions if available
        if res.predictions and len(res.predictions) > 0:
            pred = res.predictions[0]  # First face
            metrics["race"] = pred.get("race", "")
            metrics["race_conf"] = pred.get("race_confidence", 0)
            metrics["gender"] = pred.get("gender", "")
            metrics["gender_conf"] = pred.get("gender_confidence", 0)
            metrics["age"] = pred.get("age", "")
            metrics["age_conf"] = pred.get("age_confidence", 0)

        return metrics
    except Exception as e:
        logging.warning(f"Eval failed for {img_path}: {e}")
        return {}


def main():
    args = parse_args()

    # === Setup output directories ===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{timestamp}_{args.model}{'_lightning' if args.use_lightning else ''}_{args.init_type}_popt{args.p_opt_iter}_lr{args.p_opt_lr}_tlo{args.t_lo}_n{args.n_samples}"
    if args.experiment_name:
        exp_name += f"_{args.experiment_name}"

    output_dir = Path(args.output_dir) / exp_name
    baseline_dir = output_dir / "baseline"
    minority_dir = output_dir / "minority"
    modified_dir = output_dir / "modified"
    results_dir = output_dir / "results"

    for d in [baseline_dir, minority_dir, modified_dir, results_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Save config
    config_dict = vars(args)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "experiment.log"),
            logging.StreamHandler(),
        ],
    )
    logging.info(f"Output: {output_dir}")

    # === Load prompts ===
    df = pd.read_csv(args.original_csv)
    prompts = df[args.prompt_col].tolist()
    logging.info(f"Loaded {len(prompts)} prompts")

    # === Setup modifier ===
    modifier = None
    if args.enhanced_csv:
        modifier = CSVModifier(args.enhanced_csv, args.prompt_col, args.enhanced_col)
        logging.info(f"Loaded {len(modifier.lookup)} enhanced prompts")

    # === Setup generator ===
    if args.use_lightning:
        model_config = ModelConfig(
            model="sdxl_lightning", method="ddim_lightning", NFE=4, cfg_guidance=1.0
        )
    else:
        model_config = ModelConfig(
            model=args.model, method="ddim", NFE=50, cfg_guidance=7.5
        )

    popt_config = PromptOptConfig(
        enabled=True,
        p_opt_iter=args.p_opt_iter,
        p_opt_lr=args.p_opt_lr,
        t_lo=args.t_lo,
        init_type=args.init_type,
        dynamic_pr=True,
    )

    generator = MinorityGenerator(model_config=model_config, popt_config=popt_config)

    # === Setup evaluator ===
    base = Path(args.metrics_base_dir)
    evaluator = DemographicEvaluator(
        # face_model_path=str(
        #     base / "dlib_models" / "shape_predictor_5_face_landmarks.dat"
        # ),
        # fairface_model_path=str(
        #     base / "fair_face_model" / "res34_fair_align_multi_7_20190809.pt"
        # ),
        face_model_path=args.face_model_path,
        fairface_model_path=args.fairface_model_path,
        save_detected_faces=True,
    )

    # === Results storage ===
    per_image_results = []  # One row per image (per seed)
    per_prompt_results = []  # One row per prompt (aggregated)

    # === Generate and evaluate ===
    prompt_pbar = tqdm(
        range(args.start_from_row, len(prompts)), desc="Prompts", position=0
    )

    for j in prompt_pbar:
        prompt = prompts[j]
        modified_prompt = modifier.modify(prompt) if modifier else ""
        prompt_pbar.set_description(f"Prompt {j}: {prompt[:30]}...")
        logging.info(f"[{j+1}/{len(prompts)}] {prompt[:50]}...")

        # Create per-prompt results directory
        prompt_results_dir = results_dir / f"prompt_{j}"
        prompt_results_dir.mkdir(parents=True, exist_ok=True)

        sample_pbar = tqdm(
            range(args.n_samples), desc="Samples", position=1, leave=False
        )

        for i in sample_pbar:
            seed = args.seed_start + i
            img_id = f"p{j}_seed{seed}"
            sample_pbar.set_description(f"Sample {i+1}/{args.n_samples} (seed={seed})")

            # Initialize paths and result
            baseline_path = baseline_dir / f"{img_id}.png"
            minority_path = minority_dir / f"{img_id}.png"
            modified_path = modified_dir / f"{img_id}.png"

            baseline_generated = False
            minority_generated = False
            modified_generated = False

            # Generate images (with error handling)
            try:
                result = generator.generate(
                    prompt=prompt,
                    modifier=modifier,
                    seed=seed,
                    generate_baseline=True,
                    generate_minority=True,
                )

                # Save images
                if result.baseline is not None:
                    save_image(result.baseline, baseline_path)
                    baseline_generated = True
                if result.minority is not None:
                    save_image(result.minority, minority_path)
                    minority_generated = True
                if result.modified is not None:
                    save_image(result.modified, modified_path)
                    modified_generated = True

            except Exception as e:
                logging.error(f"  Generation failed for {img_id}: {e}")
                continue  # Skip to next sample

            # Evaluate each image (with error handling)
            baseline_metrics = {}
            minority_metrics = {}
            modified_metrics = {}

            try:
                if baseline_generated:
                    baseline_metrics = evaluate_single_image(
                        evaluator, baseline_path, prompt_results_dir / "baseline"
                    )
            except Exception as e:
                logging.error(f"  Baseline eval failed for {img_id}: {e}")

            try:
                if minority_generated:
                    minority_metrics = evaluate_single_image(
                        evaluator, minority_path, prompt_results_dir / "minority"
                    )
            except Exception as e:
                logging.error(f"  Minority eval failed for {img_id}: {e}")

            try:
                if modified_generated:
                    modified_metrics = evaluate_single_image(
                        evaluator, modified_path, prompt_results_dir / "modified"
                    )
            except Exception as e:
                logging.error(f"  Modified eval failed for {img_id}: {e}")

            # Build per-image row with all config + metrics
            row = {
                # Identifiers
                "prompt_idx": j,
                "seed": seed,
                "img_id": img_id,
                # Prompts
                "prompt": prompt,
                "modified_prompt": modified_prompt,
                # Config params
                "model": args.model,
                "use_lightning": args.use_lightning,
                "p_opt_iter": args.p_opt_iter,
                "p_opt_lr": args.p_opt_lr,
                "t_lo": args.t_lo,
                "init_type": args.init_type,
                "n_samples": args.n_samples,
                "seed_start": args.seed_start,
                # Image paths
                "baseline_path": str(baseline_path) if baseline_generated else "",
                "minority_path": str(minority_path) if minority_generated else "",
                "modified_path": str(modified_path) if modified_generated else "",
            }

            # Add metrics for each type
            for prefix, metrics in [
                ("baseline", baseline_metrics),
                ("minority", minority_metrics),
                ("modified", modified_metrics),
            ]:
                for k, v in metrics.items():
                    row[f"{prefix}_{k}"] = v

            per_image_results.append(row)

            # Save incrementally
            try:
                pd.DataFrame(per_image_results).to_csv(
                    results_dir / "per_image_results.csv", index=False
                )
            except Exception as e:
                logging.error(f"  Failed to save per_image_results.csv: {e}")

        # === Aggregate metrics for this prompt ===
        logging.info(f"  Evaluating prompt {j} (aggregated)...")

        prompt_row = {
            "prompt_idx": j,
            "prompt": prompt,
            "modified_prompt": modified_prompt,
        }

        # Add config
        prompt_row.update(
            {
                "model": args.model,
                "use_lightning": args.use_lightning,
                "p_opt_iter": args.p_opt_iter,
                "p_opt_lr": args.p_opt_lr,
                "t_lo": args.t_lo,
                "init_type": args.init_type,
            }
        )

        # Evaluate all images for this prompt
        for img_type, img_dir in [
            ("baseline", baseline_dir),
            ("minority", minority_dir),
            ("modified", modified_dir),
        ]:
            imgs = list(img_dir.glob(f"p{j}_*.png"))
            if imgs:
                try:
                    res = evaluator.evaluate_images(
                        images=[str(p) for p in imgs],
                        tag=f"{img_type}_{j}",
                        output_dir=prompt_results_dir / img_type,
                        progress=False,
                    )
                    res.save(prompt_results_dir / img_type, prefix=img_type)

                    prompt_row[f"{img_type}_n_images"] = res.num_images
                    prompt_row[f"{img_type}_n_faces"] = res.num_faces
                    prompt_row[f"{img_type}_face_rate"] = res.images_with_faces / max(
                        res.num_images, 1
                    )

                    if res.metrics:
                        prompt_row[f"{img_type}_bias_w_race"] = res.metrics.bias_w.get(
                            "race", 0
                        )
                        prompt_row[f"{img_type}_bias_w_gender"] = (
                            res.metrics.bias_w.get("gender", 0)
                        )
                        prompt_row[f"{img_type}_bias_w_age"] = res.metrics.bias_w.get(
                            "age", 0
                        )
                        prompt_row[f"{img_type}_ens_race"] = res.metrics.ens.get(
                            "race", 0
                        )
                        prompt_row[f"{img_type}_ens_gender"] = res.metrics.ens.get(
                            "gender", 0
                        )
                        prompt_row[f"{img_type}_ens_age"] = res.metrics.ens.get(
                            "age", 0
                        )
                        prompt_row[f"{img_type}_kl_race"] = (
                            res.metrics.kl_divergence.get("race", 0)
                        )
                        prompt_row[f"{img_type}_kl_gender"] = (
                            res.metrics.kl_divergence.get("gender", 0)
                        )
                        prompt_row[f"{img_type}_kl_age"] = (
                            res.metrics.kl_divergence.get("age", 0)
                        )
                except Exception as e:
                    logging.warning(f"  Aggregated eval failed for {img_type}: {e}")

        per_prompt_results.append(prompt_row)

        try:
            pd.DataFrame(per_prompt_results).to_csv(
                results_dir / "per_prompt_results.csv", index=False
            )
        except Exception as e:
            logging.error(f"  Failed to save per_prompt_results.csv: {e}")

        logging.info(f"  Saved results for prompt {j}")

    # === Final evaluation on all images ===
    logging.info("\nFinal evaluation on all images...")

    final_results = {}
    for img_type, img_dir in tqdm(
        [
            ("baseline", baseline_dir),
            ("minority", minority_dir),
            ("modified", modified_dir),
        ],
        desc="Final evaluation",
    ):
        try:
            if list(img_dir.glob("*.png")):
                logging.info(f"  Evaluating all {img_type} images...")
                final_results[img_type] = evaluator.evaluate_directory(
                    img_dir, tag=img_type, output_dir=results_dir, progress=True
                )
                final_results[img_type].save(results_dir, prefix=img_type)
        except Exception as e:
            logging.error(f"  Final evaluation failed for {img_type}: {e}")

    # Print comparisons
    try:
        if "baseline" in final_results and "minority" in final_results:
            print("\n" + "=" * 80)
            print("BASELINE vs MINORITY")
            print("=" * 80)
            evaluator.print_comparison(
                final_results["baseline"], final_results["minority"]
            )

        if "baseline" in final_results and "modified" in final_results:
            print("\n" + "=" * 80)
            print("BASELINE vs MODIFIED")
            print("=" * 80)
            evaluator.print_comparison(
                final_results["baseline"], final_results["modified"]
            )
    except Exception as e:
        logging.error(f"  Comparison printing failed: {e}")

    logging.info(f"\nDone! Results saved to: {output_dir}")
    logging.info(f"  - Per-image results: {results_dir / 'per_image_results.csv'}")
    logging.info(f"  - Per-prompt results: {results_dir / 'per_prompt_results.csv'}")

    return final_results


if __name__ == "__main__":
    main()
