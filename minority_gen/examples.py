#!/usr/bin/env python
"""
Example usage of the minority_gen module.

This script demonstrates:
1. Basic comparison generation
2. Using different prompt modifiers
3. Custom modifier creation
4. Batch generation
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from minority_gen import (
    MinorityGenerator,
    ModelConfig,
    PromptOptConfig,
    generate_comparison,
    GenerationResult,
)
from minority_gen.prompt_modifiers import (
    PromptModifier,
    StyleModifier,
    SuffixModifier,
    PrefixModifier,
    CompositeModifier,
    QualityBoostModifier,
    TemplateModifier,
    create_modifier,
)


def example_1_simple():
    """Simple one-liner comparison."""
    print("\n" + "="*60)
    print("Example 1: Simple Comparison")
    print("="*60)
    
    result = generate_comparison(
        prompt="Generate an image of a doctor who is smiling at the camera",
        modifier=None,  # No modifier, just baseline vs minority
        seed=42,
        model="sdxl_lightning",
        output_dir="outputs/example_1_simple",
    )
    
    print(f"Generated {sum([result.baseline is not None, result.minority is not None])} images")


def example_2_with_style_modifier():
    """Using a style modifier."""
    print("\n" + "="*60)
    print("Example 2: Style Modifier")
    print("="*60)
    
    # List available styles
    print(f"Available styles: {StyleModifier.list_styles()}")
    
    result = generate_comparison(
        prompt="A woman standing in a room",
        modifier=StyleModifier("anime"),
        seed=42,
        model="sdxl_lightning",
        output_dir="outputs/example_2_style",
    )
    
    print(f"Baseline prompt: {result.baseline_prompt}")
    print(f"Modified prompt: {result.modified_prompt}")


def example_3_custom_modifier():
    """Creating and using a custom modifier."""
    print("\n" + "="*60)
    print("Example 3: Custom Modifier")
    print("="*60)
    
    # Define a custom modifier
    class DiversityModifier(PromptModifier):
        """Modifier that adds diversity-encouraging terms."""
        
        def __init__(self, attribute: str):
            self.attribute = attribute
            
        def modify(self, prompt: str) -> str:
            # Add the attribute before the main subject
            return f"{self.attribute} {prompt}"
        
        @property
        def name(self) -> str:
            return f"diverse_{self.attribute.replace(' ', '_')}"
    
    # Use it
    modifier = DiversityModifier("elderly Asian female")
    
    result = generate_comparison(
        prompt="Generate an image of a doctor who is smiling at the camera,
        modifier=modifier,
        seed=42,
        output_dir="outputs/example_3_custom",
    )
    
    print(f"Modified prompt: {result.modified_prompt}")


def example_4_composite_modifier():
    """Chaining multiple modifiers."""
    print("\n" + "="*60)
    print("Example 4: Composite Modifier")
    print("="*60)
    
    # Chain multiple modifiers
    modifier = CompositeModifier([
        PrefixModifier("cinematic shot of"),
        SuffixModifier("golden hour lighting"),
        QualityBoostModifier("photo"),
    ], name="cinematic_golden")
    
    result = generate_comparison(
        prompt="a dog playing in the park",
        modifier=modifier,
        seed=42,
        output_dir="outputs/example_4_composite",
    )
    
    print(f"Original: {result.baseline_prompt}")
    print(f"Modified: {result.modified_prompt}")


def example_5_advanced_config():
    """Advanced configuration with custom model and optimization settings."""
    print("\n" + "="*60)
    print("Example 5: Advanced Configuration")
    print("="*60)
    
    # Custom model configuration
    model_config = ModelConfig(
        model="sdxl_lightning",
        method="ddim_lightning",
        NFE=4,
        cfg_guidance=1.0,
    )
    
    # Custom prompt optimization configuration
    popt_config = PromptOptConfig(
        enabled=True,
        p_opt_iter=5,           # 5 optimization iterations (faster)
        p_opt_lr=0.015,         # Slightly higher learning rate
        t_lo=0.0,               # Optimize at all timesteps
        dynamic_pr=True,        # Dynamic auxiliary timestep
        init_type="word",       # Initialize from word embedding
        init_word="unique",     # Start from "unique" embedding
    )
    
    # Create generator
    generator = MinorityGenerator(
        model_config=model_config,
        popt_config=popt_config,
    )
    
    # Generate
    result = generator.generate(
        prompt="A man sitting on a bench surrounded by birds",
        modifier=StyleModifier("vintage"),
        seed=42,
    )
    
    result.save("outputs/example_5_advanced")
    print("Saved with custom configuration")


def example_6_batch_generation():
    """Generate comparisons for multiple prompts."""
    print("\n" + "="*60)
    print("Example 6: Batch Generation")
    print("="*60)
    
    prompts = [
        "Generate an image of a doctor who is smiling at the camera,
        "A woman standing in a room",
        "A dog playing in the park",
        "Three people walking on a street",
    ]
    
    generator = MinorityGenerator(
        model_config=ModelConfig(model="sdxl_lightning"),
        popt_config=PromptOptConfig.fast(),
    )
    
    # Generate all comparisons
    results = generator.generate_batch(
        prompts=prompts,
        modifier=QualityBoostModifier("photo"),
        seeds=[42, 43, 44, 45],
    )
    
    # Save all results
    for i, (prompt, result) in enumerate(zip(prompts, results)):
        output_dir = Path(f"outputs/example_6_batch/prompt_{i}")
        result.save(output_dir)
        print(f"Saved prompt {i}: {prompt[:30]}...")


def example_7_template_modifier():
    """Using template-based modification."""
    print("\n" + "="*60)
    print("Example 7: Template Modifier")
    print("="*60)
    
    # Template with placeholder
    modifier = TemplateModifier(
        template="A beautiful oil painting depicting {prompt}, museum quality, ornate golden frame",
        modifier_name="museum_painting"
    )
    
    result = generate_comparison(
        prompt="a sunset over mountains",
        modifier=modifier,
        seed=42,
        output_dir="outputs/example_7_template",
    )
    
    print(f"Template result: {result.modified_prompt}")


def example_8_selective_generation():
    """Generate only specific image types."""
    print("\n" + "="*60)
    print("Example 8: Selective Generation")
    print("="*60)
    
    # Only generate minority (skip baseline)
    result = generate_comparison(
        prompt="A portrait of a scientist",
        modifier=SuffixModifier("female, lab coat"),
        seed=42,
        generate_baseline=False,  # Skip baseline
        generate_minority=True,   # Only minority
        output_dir="outputs/example_8_selective",
    )
    
    print(f"Baseline: {'Generated' if result.baseline is not None else 'Skipped'}")
    print(f"Modified: {'Generated' if result.modified is not None else 'Skipped'}")
    print(f"Minority: {'Generated' if result.minority is not None else 'Skipped'}")


def example_9_factory_function():
    """Using the modifier factory function."""
    print("\n" + "="*60)
    print("Example 9: Modifier Factory")
    print("="*60)
    
    # Create modifiers using factory
    style_mod = create_modifier("style", style="cyberpunk")
    suffix_mod = create_modifier("suffix", suffix="neon lights")
    quality_mod = create_modifier("quality", preset="art")
    
    # Combine them
    modifier = CompositeModifier([style_mod, suffix_mod, quality_mod])
    
    result = generate_comparison(
        prompt="a city street at night",
        modifier=modifier,
        seed=42,
        output_dir="outputs/example_9_factory",
    )
    
    print(f"Final prompt: {result.modified_prompt}")


def main():
    """Run selected examples."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MinorityGen Examples")
    parser.add_argument(
        "--example", 
        type=int, 
        default=1,
        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9],
        help="Which example to run (1-9)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all examples"
    )
    
    args = parser.parse_args()
    
    examples = {
        1: example_1_simple,
        2: example_2_with_style_modifier,
        3: example_3_custom_modifier,
        4: example_4_composite_modifier,
        5: example_5_advanced_config,
        6: example_6_batch_generation,
        7: example_7_template_modifier,
        8: example_8_selective_generation,
        9: example_9_factory_function,
    }
    
    if args.all:
        for num, func in examples.items():
            try:
                func()
            except Exception as e:
                print(f"Example {num} failed: {e}")
    else:
        examples[args.example]()


if __name__ == "__main__":
    main()
