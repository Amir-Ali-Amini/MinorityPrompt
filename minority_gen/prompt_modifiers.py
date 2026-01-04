"""
Prompt modifier base class and implementations.

Users can create custom prompt modifiers by subclassing PromptModifier
and implementing the modify() method.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from dataclasses import dataclass


class PromptModifier(ABC):
    """
    Base class for prompt modification modules.
    
    Subclass this and implement modify() to create custom prompt transformations.
    The modifier is applied BEFORE generation, transforming the text prompt.
    
    Example:
        class MyModifier(PromptModifier):
            def __init__(self, style: str):
                self.style = style
                
            def modify(self, prompt: str) -> str:
                return f"{prompt}, {self.style} style"
                
            @property
            def name(self) -> str:
                return f"style_{self.style}"
    """
    
    @abstractmethod
    def modify(self, prompt: str) -> str:
        """
        Transform the input prompt.
        
        Args:
            prompt: Original text prompt
            
        Returns:
            Modified text prompt
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique identifier for this modifier.
        Used in output filenames and logging.
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"


# =============================================================================
# Example Implementations
# =============================================================================

class SuffixModifier(PromptModifier):
    """Add a suffix to the prompt."""
    
    def __init__(self, suffix: str, modifier_name: Optional[str] = None):
        """
        Args:
            suffix: Text to append to prompt
            modifier_name: Optional custom name (defaults to suffix)
        """
        self.suffix = suffix
        self._name = modifier_name or suffix.replace(" ", "_")[:20]
    
    def modify(self, prompt: str) -> str:
        return f"{prompt}, {self.suffix}"
    
    @property
    def name(self) -> str:
        return f"suffix_{self._name}"


class PrefixModifier(PromptModifier):
    """Add a prefix to the prompt."""
    
    def __init__(self, prefix: str, modifier_name: Optional[str] = None):
        """
        Args:
            prefix: Text to prepend to prompt
            modifier_name: Optional custom name (defaults to prefix)
        """
        self.prefix = prefix
        self._name = modifier_name or prefix.replace(" ", "_")[:20]
    
    def modify(self, prompt: str) -> str:
        return f"{self.prefix} {prompt}"
    
    @property
    def name(self) -> str:
        return f"prefix_{self._name}"


class StyleModifier(PromptModifier):
    """Apply a predefined artistic style."""
    
    STYLES = {
        "anime": "anime style, vibrant colors, detailed",
        "photorealistic": "photorealistic, 8k, ultra detailed, professional photography",
        "oil_painting": "oil painting, classical art style, textured brushstrokes",
        "watercolor": "watercolor painting, soft edges, artistic",
        "sketch": "pencil sketch, black and white, detailed linework",
        "cyberpunk": "cyberpunk style, neon lights, futuristic, dark atmosphere",
        "vintage": "vintage photograph, sepia tones, nostalgic, 1950s style",
        "minimalist": "minimalist, simple, clean lines, modern design",
        "fantasy": "fantasy art, magical, ethereal, detailed illustration",
        "comic": "comic book style, bold lines, cel shading",
    }
    
    def __init__(self, style: str):
        """
        Args:
            style: Style name (see StyleModifier.STYLES for options)
        """
        if style not in self.STYLES:
            available = ", ".join(self.STYLES.keys())
            raise ValueError(f"Unknown style '{style}'. Available: {available}")
        self.style = style
        self.style_prompt = self.STYLES[style]
    
    def modify(self, prompt: str) -> str:
        return f"{prompt}, {self.style_prompt}"
    
    @property
    def name(self) -> str:
        return f"style_{self.style}"
    
    @classmethod
    def list_styles(cls) -> List[str]:
        """Return list of available style names."""
        return list(cls.STYLES.keys())


class NegativePromptModifier(PromptModifier):
    """
    Modify by setting a negative prompt.
    
    Note: This is a special modifier that affects null_prompt instead of prompt.
    The generator handles this specially.
    """
    
    def __init__(self, negative_prompt: str, modifier_name: Optional[str] = None):
        """
        Args:
            negative_prompt: Text describing what to avoid
            modifier_name: Optional custom name
        """
        self.negative_prompt = negative_prompt
        self._name = modifier_name or "negative"
    
    def modify(self, prompt: str) -> str:
        # Return prompt unchanged; the generator will use self.negative_prompt
        return prompt
    
    @property
    def name(self) -> str:
        return f"neg_{self._name}"


class QualityBoostModifier(PromptModifier):
    """Add quality enhancement terms."""
    
    PRESETS = {
        "basic": "high quality, detailed",
        "photo": "professional photography, 8k resolution, sharp focus, detailed",
        "art": "masterpiece, best quality, highly detailed, artstation",
        "portrait": "professional portrait, studio lighting, sharp focus, high detail",
    }
    
    def __init__(self, preset: str = "basic"):
        if preset not in self.PRESETS:
            available = ", ".join(self.PRESETS.keys())
            raise ValueError(f"Unknown preset '{preset}'. Available: {available}")
        self.preset = preset
        self.quality_terms = self.PRESETS[preset]
    
    def modify(self, prompt: str) -> str:
        return f"{prompt}, {self.quality_terms}"
    
    @property
    def name(self) -> str:
        return f"quality_{self.preset}"


class CompositeModifier(PromptModifier):
    """Chain multiple modifiers together."""
    
    def __init__(self, modifiers: List[PromptModifier], name: Optional[str] = None):
        """
        Args:
            modifiers: List of modifiers to apply in order
            name: Optional custom name (defaults to joined modifier names)
        """
        self.modifiers = modifiers
        self._name = name or "+".join(m.name for m in modifiers)
    
    def modify(self, prompt: str) -> str:
        result = prompt
        for modifier in self.modifiers:
            result = modifier.modify(result)
        return result
    
    @property
    def name(self) -> str:
        return self._name


class TemplateModifier(PromptModifier):
    """Apply a template with placeholder."""
    
    def __init__(self, template: str, modifier_name: str):
        """
        Args:
            template: Template string with {prompt} placeholder
            modifier_name: Name for this modifier
            
        Example:
            TemplateModifier(
                template="A beautiful photograph of {prompt}, award winning",
                modifier_name="photo_template"
            )
        """
        if "{prompt}" not in template:
            raise ValueError("Template must contain {prompt} placeholder")
        self.template = template
        self._name = modifier_name
    
    def modify(self, prompt: str) -> str:
        return self.template.format(prompt=prompt)
    
    @property
    def name(self) -> str:
        return self._name


class AttributeModifier(PromptModifier):
    """Add specific attributes to subjects in the prompt."""
    
    def __init__(self, attributes: Dict[str, str], modifier_name: str):
        """
        Args:
            attributes: Dict mapping subjects to attributes
            modifier_name: Name for this modifier
            
        Example:
            AttributeModifier(
                attributes={"chef": "female Asian", "dog": "golden retriever"},
                modifier_name="diverse_attrs"
            )
        """
        self.attributes = attributes
        self._name = modifier_name
    
    def modify(self, prompt: str) -> str:
        result = prompt
        for subject, attribute in self.attributes.items():
            # Simple replacement - could be made smarter with NLP
            if subject in result.lower():
                result = result.replace(subject, f"{attribute} {subject}")
                result = result.replace(subject.capitalize(), f"{attribute} {subject}")
        return result
    
    @property
    def name(self) -> str:
        return self._name


# =============================================================================
# Convenience functions
# =============================================================================

def create_modifier(modifier_type: str, **kwargs) -> PromptModifier:
    """
    Factory function to create modifiers by name.
    
    Args:
        modifier_type: One of "suffix", "prefix", "style", "quality", "template"
        **kwargs: Arguments passed to the modifier constructor
        
    Returns:
        PromptModifier instance
        
    Example:
        modifier = create_modifier("style", style="anime")
        modifier = create_modifier("suffix", suffix="beautiful sunset")
    """
    constructors = {
        "suffix": SuffixModifier,
        "prefix": PrefixModifier,
        "style": StyleModifier,
        "quality": QualityBoostModifier,
        "template": TemplateModifier,
        "negative": NegativePromptModifier,
        "attribute": AttributeModifier,
    }
    
    if modifier_type not in constructors:
        available = ", ".join(constructors.keys())
        raise ValueError(f"Unknown modifier type '{modifier_type}'. Available: {available}")
    
    return constructors[modifier_type](**kwargs)
