"""
Configuration dataclasses for MinorityPrompt generation.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class ModelConfig:
    """Configuration for the diffusion model."""
    
    # Model selection
    model: Literal["sd15", "sd20", "sdxl", "sdxl_lightning"] = "sdxl_lightning"
    
    # Sampling method
    method: Literal["ddim", "ddim_cfg++", "ddim_lightning", "ddim_cfg++_lightning"] = "ddim_lightning"
    
    # Number of Function Evaluations (sampling steps)
    # - sd15/sd20/sdxl: typically 50
    # - sdxl_lightning: 4 or 8
    NFE: int = 4
    
    # Classifier-Free Guidance scale
    # - sd15/sd20/sdxl: typically 7.5
    # - sdxl_lightning: must be 1.0 (CFG baked into distillation)
    cfg_guidance: float = 1.0
    
    # Device
    device: str = "cuda"
    
    # Lightning checkpoint path (only for sdxl_lightning)
    lightning_ckpt: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Auto-set method based on model
        if self.model == "sdxl_lightning" and "lightning" not in self.method:
            self.method = "ddim_lightning"
            
        # Enforce cfg_guidance=1.0 for lightning models
        if "lightning" in self.model and self.cfg_guidance != 1.0:
            print(f"Warning: Setting cfg_guidance=1.0 for {self.model} (required)")
            self.cfg_guidance = 1.0
            
        # Auto-set lightning checkpoint path
        if self.model == "sdxl_lightning" and self.lightning_ckpt is None:
            self.lightning_ckpt = f"ckpt/sdxl_lightning_{self.NFE}step_unet.safetensors"


@dataclass
class PromptOptConfig:
    """Configuration for MinorityPrompt optimization."""
    
    # Enable prompt optimization
    enabled: bool = True
    
    # Auxiliary timestep ratio for loss computation
    # Controls noise level for reconstruction comparison
    # Range: 0.0-1.0, default 0.75
    p_ratio: float = 0.75
    
    # Number of inner optimization iterations per sampling step
    p_opt_iter: int = 10
    
    # Learning rate for embedding optimization
    p_opt_lr: float = 0.01
    
    # Threshold for starting optimization (fraction of total timesteps)
    # 0.0 = optimize at all timesteps
    # 0.9 = only optimize at first 10% of denoising (early steps)
    t_lo: float = 0.0
    
    # Placeholder token configuration
    placeholder_string: str = "*_0"
    num_opt_tokens: int = 1
    placeholder_position: Literal["start", "end"] = "end"
    
    # Initialization type for placeholder embeddings
    # - "default": random from resize
    # - "word": initialize from init_word embedding
    # - "gaussian": sample from full covariance of vocab embeddings
    # - "gaussian_white": sample from diagonal covariance
    init_type: Literal["default", "word", "gaussian", "gaussian_white"] = "default"
    init_word: str = ""
    init_gau_scale: float = 1.0
    init_rand_vocab: bool = False
    
    # Dynamic p_ratio adjustment based on current timestep
    dynamic_pr: bool = True
    
    # Revert to base prompt between optimization steps
    base_prompt_after_popt: bool = False
    
    # Optimization interval (every N steps)
    inter_rate: int = 1
    
    # Learning rate decay
    lr_decay_rate: float = 0.0
    
    # Stop-gradient term weight for stability
    sg_lambda: float = 1.0
    
    # Diversity mode (for batch generation)
    popt_diverse: bool = False
    
    def to_dict(self) -> dict:
        """Convert to dictionary for solver compatibility."""
        return {
            "prompt_opt": self.enabled,
            "p_ratio": self.p_ratio,
            "p_opt_iter": self.p_opt_iter,
            "p_opt_lr": self.p_opt_lr,
            "t_lo": self.t_lo,
            "placeholder_string": self.placeholder_string,
            "num_opt_tokens": self.num_opt_tokens,
            "placeholder_position": self.placeholder_position,
            "init_type": self.init_type,
            "init_word": self.init_word,
            "init_gau_scale": self.init_gau_scale,
            "init_rand_vocab": self.init_rand_vocab,
            "dynamic_pr": self.dynamic_pr,
            "base_prompt_after_popt": self.base_prompt_after_popt,
            "inter_rate": self.inter_rate,
            "lr_decay_rate": self.lr_decay_rate,
            "sg_lambda": self.sg_lambda,
            "popt_diverse": self.popt_diverse,
        }
    
    @classmethod
    def disabled(cls) -> "PromptOptConfig":
        """Create a disabled configuration (for baseline)."""
        return cls(enabled=False)
    
    @classmethod
    def fast(cls) -> "PromptOptConfig":
        """Fast configuration with fewer iterations."""
        return cls(
            enabled=True,
            p_opt_iter=3,
            t_lo=0.0,
            dynamic_pr=True,
        )
    
    @classmethod
    def quality(cls) -> "PromptOptConfig":
        """High-quality configuration with more iterations."""
        return cls(
            enabled=True,
            p_opt_iter=10,
            t_lo=0.0,
            dynamic_pr=True,
            p_opt_lr=0.01,
        )


@dataclass
class GenerationConfig:
    """Complete configuration for image generation."""
    
    # The text prompt
    prompt: str = ""
    
    # Negative prompt (typically empty)
    null_prompt: str = ""
    
    # Random seed for reproducibility
    seed: int = 42
    
    # Output image size (for SDXL models)
    target_size: tuple = (1024, 1024)
    
    # Model configuration
    model_config: ModelConfig = field(default_factory=ModelConfig)
    
    # Prompt optimization configuration
    popt_config: PromptOptConfig = field(default_factory=PromptOptConfig)
