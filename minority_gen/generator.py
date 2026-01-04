"""
Main generator module for MinorityPrompt comparison.

This module provides a simple interface to generate:
1. Baseline image (no modifications)
2. Image with custom prompt modifier (optional)
3. Image with MinorityPrompt optimization
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Union
from dataclasses import dataclass, field
import copy

import torch
from torchvision.utils import save_image
from munch import munchify

# Add parent directory to path for imports
_SCRIPT_DIR = Path(__file__).parent.absolute()
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from latent_diffusion import get_solver as get_solver_sd
from latent_sdxl import get_solver as get_solver_sdxl

from .config import ModelConfig, PromptOptConfig, GenerationConfig
from .prompt_modifiers import PromptModifier, NegativePromptModifier


@dataclass
class GenerationResult:
    """Container for generation results."""
    
    # Image tensors [1, 3, H, W] in range [0, 1]
    baseline: Optional[torch.Tensor] = None
    modified: Optional[torch.Tensor] = None
    minority: Optional[torch.Tensor] = None
    
    # Prompts used
    baseline_prompt: str = ""
    modified_prompt: str = ""
    minority_prompt: str = ""
    
    # Modifier info
    modifier_name: Optional[str] = None
    
    # Metadata
    seed: int = 0
    model: str = ""
    
    def save(self, output_dir: Union[str, Path], prefix: str = "") -> Dict[str, Path]:
        """
        Save all generated images to directory.
        
        Args:
            output_dir: Directory to save images
            prefix: Optional prefix for filenames
            
        Returns:
            Dictionary mapping result type to saved file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved = {}
        prefix = f"{prefix}_" if prefix else ""
        
        if self.baseline is not None:
            path = output_dir / f"{prefix}baseline.png"
            save_image(self.baseline, path)
            saved["baseline"] = path
            
        if self.modified is not None:
            modifier_suffix = f"_{self.modifier_name}" if self.modifier_name else "_modified"
            path = output_dir / f"{prefix}modified{modifier_suffix}.png"
            save_image(self.modified, path)
            saved["modified"] = path
            
        if self.minority is not None:
            path = output_dir / f"{prefix}minority.png"
            save_image(self.minority, path)
            saved["minority"] = path
            
        return saved
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "baseline_prompt": self.baseline_prompt,
            "modified_prompt": self.modified_prompt,
            "minority_prompt": self.minority_prompt,
            "modifier_name": self.modifier_name,
            "seed": self.seed,
            "model": self.model,
            "has_baseline": self.baseline is not None,
            "has_modified": self.modified is not None,
            "has_minority": self.minority is not None,
        }


class MinorityGenerator:
    """
    Main class for generating comparison images.
    
    Generates up to 3 images per prompt:
    1. Baseline: Standard generation without any modifications
    2. Modified: Generation with custom prompt modifier applied
    3. Minority: Generation with MinorityPrompt optimization
    
    Example:
        from minority_gen import MinorityGenerator, ModelConfig, PromptOptConfig
        from minority_gen.prompt_modifiers import StyleModifier
        
        # Initialize generator
        generator = MinorityGenerator(
            model_config=ModelConfig(model="sdxl_lightning", NFE=4)
        )
        
        # Create modifier (optional)
        modifier = StyleModifier("anime")
        
        # Generate comparison
        result = generator.generate(
            prompt="A portrait of a chef",
            modifier=modifier,  # None to skip modified image
            seed=42
        )
        
        # Save results
        result.save("outputs/comparison")
    """
    
    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        popt_config: Optional[PromptOptConfig] = None,
        device: str = "cuda",
    ):
        """
        Initialize the generator.
        
        Args:
            model_config: Model configuration (defaults to SDXL-Lightning)
            popt_config: Prompt optimization config (defaults to fast preset)
            device: Device to run on
        """
        self.model_config = model_config or ModelConfig()
        self.model_config.device = device
        
        self.popt_config = popt_config or PromptOptConfig.fast()
        self.device = device
        
        # Solver will be created lazily
        self._solver = None
        self._solver_config = None
        
    def _get_solver(self):
        """Get or create the solver (lazy initialization)."""
        solver_config = munchify({"num_sampling": self.model_config.NFE})
        
        # Check if we need to recreate solver
        if self._solver is not None and self._solver_config == solver_config:
            return self._solver
            
        model = self.model_config.model
        method = self.model_config.method
        
        if model in ["sdxl", "sdxl_lightning"]:
            if model == "sdxl_lightning":
                self._solver = get_solver_sdxl(
                    method,
                    solver_config=solver_config,
                    device=self.device,
                    light_model_ckpt=self.model_config.lightning_ckpt,
                )
            else:
                self._solver = get_solver_sdxl(
                    method,
                    solver_config=solver_config,
                    device=self.device,
                )
        else:
            model_key = (
                "botp/stable-diffusion-v1-5" if model == "sd15" 
                else "stabilityai/stable-diffusion-2-base"
            )
            self._solver = get_solver_sd(
                method,
                solver_config=solver_config,
                model_key=model_key,
                device=self.device,
            )
            
        self._solver_config = solver_config
        return self._solver
    
    def _set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    def _generate_single(
        self,
        prompt: str,
        null_prompt: str,
        popt_kwargs: dict,
        seed: int,
    ) -> torch.Tensor:
        """Generate a single image."""
        self._set_seed(seed)
        
        # Need fresh solver for each generation to reset tokenizer
        solver = self._get_solver()
        
        model = self.model_config.model
        cfg = self.model_config.cfg_guidance
        
        if model in ["sdxl", "sdxl_lightning"]:
            result = solver.sample(
                prompt1=[null_prompt, prompt],
                prompt2=[null_prompt, prompt],
                cfg_guidance=cfg,
                target_size=(1024, 1024),
                popt_kwargs=popt_kwargs,
            )
        else:
            result = solver.sample(
                prompt=[null_prompt, prompt],
                cfg_guidance=cfg,
                popt_kwargs=popt_kwargs,
            )
            
        return result
    
    def generate(
        self,
        prompt: str,
        modifier: Optional[PromptModifier] = None,
        seed: int = 42,
        null_prompt: str = "",
        generate_baseline: bool = True,
        generate_minority: bool = True,
        popt_config_override: Optional[PromptOptConfig] = None,
    ) -> GenerationResult:
        """
        Generate comparison images.
        
        Args:
            prompt: The text prompt to generate from
            modifier: Optional prompt modifier for the "modified" image
            seed: Random seed for reproducibility
            null_prompt: Negative prompt (typically empty)
            generate_baseline: Whether to generate baseline image
            generate_minority: Whether to generate minority image
            popt_config_override: Override prompt optimization config
            
        Returns:
            GenerationResult containing all generated images
        """
        result = GenerationResult(
            seed=seed,
            model=self.model_config.model,
            baseline_prompt=prompt,
        )
        
        # Configuration for no optimization
        baseline_popt = PromptOptConfig.disabled().to_dict()
        
        # Configuration for minority optimization
        popt_config = popt_config_override or self.popt_config
        minority_popt = popt_config.to_dict()
        
        # Handle negative prompt from modifier
        effective_null_prompt = null_prompt
        if modifier and isinstance(modifier, NegativePromptModifier):
            effective_null_prompt = modifier.negative_prompt
        
        # 1. Generate baseline
        if generate_baseline:
            print(f"Generating baseline: '{prompt}'")
            result.baseline = self._generate_single(
                prompt=prompt,
                null_prompt=null_prompt,
                popt_kwargs=baseline_popt,
                seed=seed,
            )
            result.baseline_prompt = prompt
            
        # 2. Generate with modifier
        if modifier is not None:
            modified_prompt = modifier.modify(prompt)
            print(f"Generating modified ({modifier.name}): '{modified_prompt}'")
            
            # Use effective null prompt (may be modified)
            result.modified = self._generate_single(
                prompt=modified_prompt,
                null_prompt=effective_null_prompt,
                popt_kwargs=baseline_popt,  # No optimization, just prompt change
                seed=seed,
            )
            result.modified_prompt = modified_prompt
            result.modifier_name = modifier.name
            
        # 3. Generate with MinorityPrompt
        if generate_minority:
            print(f"Generating minority: '{prompt}'")
            result.minority = self._generate_single(
                prompt=prompt,
                null_prompt=null_prompt,
                popt_kwargs=minority_popt,
                seed=seed,
            )
            result.minority_prompt = prompt  # Same prompt, different method
            
        return result
    
    def generate_batch(
        self,
        prompts: List[str],
        modifier: Optional[PromptModifier] = None,
        seeds: Optional[List[int]] = None,
        **kwargs,
    ) -> List[GenerationResult]:
        """
        Generate comparisons for multiple prompts.
        
        Args:
            prompts: List of text prompts
            modifier: Optional modifier applied to all prompts
            seeds: Optional list of seeds (defaults to [42, 43, 44, ...])
            **kwargs: Additional arguments passed to generate()
            
        Returns:
            List of GenerationResult objects
        """
        if seeds is None:
            seeds = list(range(42, 42 + len(prompts)))
        elif len(seeds) != len(prompts):
            raise ValueError("Number of seeds must match number of prompts")
            
        results = []
        for i, (prompt, seed) in enumerate(zip(prompts, seeds)):
            print(f"\n[{i+1}/{len(prompts)}] Processing: {prompt[:50]}...")
            result = self.generate(prompt=prompt, modifier=modifier, seed=seed, **kwargs)
            results.append(result)
            
        return results


def generate_comparison(
    prompt: str,
    modifier: Optional[PromptModifier] = None,
    seed: int = 42,
    model: str = "sdxl_lightning",
    output_dir: Optional[Union[str, Path]] = None,
    **kwargs,
) -> GenerationResult:
    """
    Convenience function for quick comparison generation.
    
    Args:
        prompt: Text prompt
        modifier: Optional prompt modifier
        seed: Random seed
        model: Model name ("sd15", "sd20", "sdxl", "sdxl_lightning")
        output_dir: If provided, save images to this directory
        **kwargs: Additional arguments passed to MinorityGenerator.generate()
        
    Returns:
        GenerationResult with generated images
        
    Example:
        from minority_gen import generate_comparison
        from minority_gen.prompt_modifiers import StyleModifier
        
        result = generate_comparison(
            prompt="A portrait of a chef",
            modifier=StyleModifier("anime"),
            seed=42,
            output_dir="outputs/chef_comparison"
        )
    """
    # Configure model
    if model == "sdxl_lightning":
        model_config = ModelConfig(
            model="sdxl_lightning",
            method="ddim_lightning",
            NFE=4,
            cfg_guidance=1.0,
        )
    elif model == "sdxl":
        model_config = ModelConfig(
            model="sdxl",
            method="ddim",
            NFE=50,
            cfg_guidance=7.5,
        )
    elif model == "sd15":
        model_config = ModelConfig(
            model="sd15",
            method="ddim",
            NFE=50,
            cfg_guidance=7.5,
        )
    elif model == "sd20":
        model_config = ModelConfig(
            model="sd20",
            method="ddim",
            NFE=50,
            cfg_guidance=7.5,
        )
    else:
        raise ValueError(f"Unknown model: {model}")
    
    # Create generator and generate
    generator = MinorityGenerator(model_config=model_config)
    result = generator.generate(prompt=prompt, modifier=modifier, seed=seed, **kwargs)
    
    # Save if output_dir provided
    if output_dir:
        saved = result.save(output_dir)
        print(f"Saved images to: {output_dir}")
        for name, path in saved.items():
            print(f"  - {name}: {path}")
            
    return result
