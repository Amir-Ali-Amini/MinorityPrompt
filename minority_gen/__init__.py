"""
MinorityPrompt Generation Module

A simple interface for generating comparison images:
1. Baseline (standard generation)
2. Modified (with custom prompt modifier)
3. Minority (with MinorityPrompt optimization)

Quick Start:
    from minority_gen import generate_comparison
    from minority_gen.prompt_modifiers import StyleModifier

    # Simple usage
    result = generate_comparison(
        prompt="Generate an image of a doctor who is smiling at the camera",
        modifier=StyleModifier("anime"),  # or None to skip
        seed=42,
        output_dir="outputs/"
    )

    # Access images directly
    baseline_tensor = result.baseline
    modified_tensor = result.modified
    minority_tensor = result.minority

Advanced Usage:
    from minority_gen import MinorityGenerator, ModelConfig, PromptOptConfig
    from minority_gen.prompt_modifiers import CompositeModifier, SuffixModifier, QualityBoostModifier

    # Custom configuration
    generator = MinorityGenerator(
        model_config=ModelConfig(
            model="sdxl_lightning",
            NFE=4,
            cfg_guidance=1.0,
        ),
        popt_config=PromptOptConfig(
            enabled=True,
            p_opt_iter=5,
            t_lo=0.0,
            dynamic_pr=True,
        )
    )

    # Chain multiple modifiers
    modifier = CompositeModifier([
        SuffixModifier("golden hour lighting"),
        QualityBoostModifier("photo"),
    ])

    result = generator.generate(
        prompt="A woman walking in a park",
        modifier=modifier,
        seed=123,
    )

    result.save("outputs/park_comparison")

Custom Prompt Modifier:
    from minority_gen.prompt_modifiers import PromptModifier

    class MyModifier(PromptModifier):
        def __init__(self, attribute: str):
            self.attribute = attribute

        def modify(self, prompt: str) -> str:
            return f"{self.attribute} {prompt}"

        @property
        def name(self) -> str:
            return f"my_{self.attribute}"

    # Use it
    result = generate_comparison(
        prompt="A dog playing",
        modifier=MyModifier("happy"),
    )
"""

from .config import ModelConfig, PromptOptConfig, GenerationConfig
from .generator import MinorityGenerator, GenerationResult, generate_comparison
from .prompt_modifiers import (
    PromptModifier,
    SuffixModifier,
    PrefixModifier,
    StyleModifier,
    NegativePromptModifier,
    QualityBoostModifier,
    CompositeModifier,
    TemplateModifier,
    AttributeModifier,
    create_modifier,
)

__all__ = [
    # Main classes
    "MinorityGenerator",
    "GenerationResult",
    # Convenience function
    "generate_comparison",
    # Configuration
    "ModelConfig",
    "PromptOptConfig",
    "GenerationConfig",
    # Prompt modifiers
    "PromptModifier",
    "SuffixModifier",
    "PrefixModifier",
    "StyleModifier",
    "NegativePromptModifier",
    "QualityBoostModifier",
    "CompositeModifier",
    "TemplateModifier",
    "AttributeModifier",
    "create_modifier",
]

__version__ = "0.1.0"

# Evaluation module (optional import - requires dlib and FairFace model)
try:
    from . import evaluation
    from .evaluation import (
        DemographicEvaluator,
        EvaluationResult,
        quick_evaluate,
        BiasMetrics,
    )

    __all__.extend(
        [
            "evaluation",
            "DemographicEvaluator",
            "EvaluationResult",
            "quick_evaluate",
            "BiasMetrics",
        ]
    )
except ImportError:
    pass  # dlib or other dependencies not available
