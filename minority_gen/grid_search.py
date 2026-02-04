import random
import math
from pathlib import Path
from datetime import datetime
from tqdm import tqdm


# Parameter specifications with ranges and scales
# PARAM_SPECS = {
#     "p_opt_lr": {"low": 0.001, "high": 0.1, "scale": "log"},
#     "p_opt_iter": {"low": 5, "high": 30, "scale": "linear", "dtype": int},
#     "t_lo": {"low": 0.0, "high": 0.5, "scale": "linear"},
#     "init_type": {"choices": ["default", "random", "learned"], "scale": "categorical"},
#     "seed_plus": {"low": 0, "high": 1000, "scale": "linear", "dtype": int},
# }
PARAM_SPECS = {
    "p_opt_lr": {"low": 0.001, "high": 0.1, "scale": "log"},
    "p_opt_iter": {"low": 10, "high": 10, "scale": "linear", "dtype": int},
    "t_lo": {"low": 0.0, "high": 0.5, "scale": "linear"},
    "init_type": {"choices": ["default"], "scale": "categorical"},
    "seed_plus": {"low": 0, "high": 0, "scale": "linear", "dtype": int},
}


def sample_param(name: str) -> any:
    """Sample a random value for a parameter using appropriate scale."""
    spec = PARAM_SPECS[name]

    if spec["scale"] == "log":
        # Log-uniform sampling
        log_low = math.log(spec["low"])
        log_high = math.log(spec["high"])
        value = math.exp(random.uniform(log_low, log_high))
    elif spec["scale"] == "linear":
        value = random.uniform(spec["low"], spec["high"])
        if spec.get("dtype") == int:
            value = int(round(value))
    elif spec["scale"] == "categorical":
        value = random.choice(spec["choices"])

    return value


def sample_all_params() -> dict:
    """Sample random values for all parameters."""
    return {name: sample_param(name) for name in PARAM_SPECS}


def get_model_config(model: str):
    """Return appropriate ModelConfig for each model type."""
    from ..minority_gen import ModelConfig

    configs = {
        "sdxl_lightning": ModelConfig(
            model="sdxl_lightning",
            method="ddim_lightning",
            NFE=4,
            cfg_guidance=1.0,
        ),
        "sdxl": ModelConfig(
            model="sdxl",
            method="ddim",
            NFE=50,
            cfg_guidance=7.5,
        ),
        "sd15": ModelConfig(
            model="sd15",
            method="ddim",
            NFE=50,
            cfg_guidance=7.5,
        ),
    }

    if model not in configs:
        raise ValueError(f"Unknown model: {model}")
    return configs[model]


def run_single_config(
    model: str,
    params: dict,
    n_samples: int,
    prompts: list[str],
    output_base: Path,
):
    """Run generation for a single parameter configuration (no evaluation)."""
    from torchvision.utils import save_image
    from ..minority_gen import MinorityGenerator, PromptOptConfig
    from .prompt_modifiers import CompositeModifier, SharifModifier

    param_str = (
        f"{model}_"
        f"iter{params['p_opt_iter']}_"
        f"lr{params['p_opt_lr']:.4f}_"
        f"tlo{params['t_lo']:.2f}_"
        f"init{params['init_type']}_"
        f"seed{params['seed_plus']}"
    )
    output_dir = output_base / param_str

    baseline_dir = output_dir / "baseline"
    minority_dir = output_dir / "minority"
    modified_dir = output_dir / "modified"

    baseline_dir.mkdir(parents=True, exist_ok=True)
    minority_dir.mkdir(parents=True, exist_ok=True)
    modified_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.txt", "w") as f:
        f.write(f"model: {model}\n")
        for k, v in params.items():
            f.write(f"{k}: {v}\n")

    model_config = get_model_config(model)
    popt_config = PromptOptConfig(
        enabled=True,
        p_opt_iter=params["p_opt_iter"],
        p_opt_lr=params["p_opt_lr"],
        t_lo=params["t_lo"],
        dynamic_pr=True,
        init_type=params["init_type"],
    )

    generator = MinorityGenerator(
        model_config=model_config,
        popt_config=popt_config,
    )

    modifier = CompositeModifier([SharifModifier()])

    for j, prompt in enumerate(prompts):
        print(f"    [{j+1}/{len(prompts)}] {prompt[:50]}...")
        for i in tqdm(range(n_samples), desc=f"    Prompt {j+1}"):
            result = generator.generate(
                prompt=prompt,
                # modifier=modifier,
                seed=42 + i + params["seed_plus"],
                # generate_baseline=True,
                generate_minority=True,
            )

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

    return output_dir


def random_search(
    models: list[str] = None,
    n_configs: int = 25,
    n_samples: int = 5 and 1,
    prompts: list[str] = None,
    seed: int = 42,
):
    """
    Random search: sample n_configs random points in the parameter space.

    Args:
        models: Models to test
        n_configs: Number of random configurations per model
        n_samples: Images per prompt per config
        prompts: List of prompts
        seed: Random seed for reproducibility
    """
    if models is None:
        models = ["sdxl", "sdxl_lightning", "sd15"]

    if prompts is None:
        prompts = [
            "Generate an image of a doctor who is smiling at the camera",
            "Generate an image of a nurse who is smiling at the camera",
        ]

    # Print parameter ranges
    print("Parameter Ranges:")
    print("-" * 60)
    for name, spec in PARAM_SPECS.items():
        if spec["scale"] == "categorical":
            print(f"  {name}: {spec['choices']} (categorical)")
        else:
            print(f"  {name}: [{spec['low']}, {spec['high']}] ({spec['scale']} scale)")
    print("-" * 60)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path(f"outputs/random_search_{timestamp}")
    output_base.mkdir(parents=True, exist_ok=True)

    # Save search info
    with open(output_base / "search_info.txt", "w") as f:
        f.write(f"seed: {seed}\n")
        f.write(f"n_configs per model: {n_configs}\n")
        f.write(f"models: {models}\n\n")
        f.write("Parameter ranges:\n")
        for name, spec in PARAM_SPECS.items():
            f.write(f"  {name}: {spec}\n")

    log_path = output_base / "runs.txt"

    total_runs = len(models) * n_configs
    run_idx = 0

    for model in models:
        print(f"\n{'='*70}")
        print(f"Model: {model}")
        print(f"{'='*70}")

        # Set seed per model for reproducibility
        random.seed(seed + hash(model) % 10000)

        for config_idx in range(n_configs):
            run_idx += 1
            params = sample_all_params()

            print(f"\n  [Run {run_idx}/{total_runs}]")
            print(f"    p_opt_lr:   {params['p_opt_lr']:.6f}")
            print(f"    p_opt_iter: {params['p_opt_iter']}")
            print(f"    t_lo:       {params['t_lo']:.4f}")
            print(f"    init_type:  {params['init_type']}")
            print(f"    seed_plus:  {params['seed_plus']}")

            with open(log_path, "a") as f:
                f.write(f"Run {run_idx}: {model} - {params}\n")

            output_dir = run_single_config(
                model=model,
                params=params,
                n_samples=n_samples,
                prompts=prompts,
                output_base=output_base / model,
            )

            print(f"    Saved: {output_dir}")

    print(f"\n{'='*70}")
    print(f"Random search complete!")
    print(f"Results: {output_base}")
    print(f"Total runs: {total_runs}")
    print(f"{'='*70}")

    return output_base


if __name__ == "__main__":
    random_search(
        # models=["sdxl", "sdxl_lightning", "sd15"],
        models=["sd15"],
        n_configs=25,
        n_samples=1,
        seed=42,
    )
