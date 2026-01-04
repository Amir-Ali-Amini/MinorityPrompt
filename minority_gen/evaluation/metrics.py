"""
Bias and diversity metrics for demographic evaluation.

This module implements several metrics for measuring demographic bias
and diversity in generated images:

1. Bias-W: Population-level bias measuring deviation from uniform distribution
2. Bias-P: Per-image bias for multi-face images  
3. ENS: Effective Number of Species (diversity metric)
4. KL Divergence: Kullback-Leibler divergence from reference distribution

References:
- Bias metrics from demographic analysis literature
- ENS from ecology/biodiversity measurement
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from itertools import product

import numpy as np
import pandas as pd

from .constants import ATTRIBUTES_DICT, ATTRIBUTE_COMBINATIONS


@dataclass
class BiasMetrics:
    """Container for bias metric results."""
    
    # Bias-W: population-level bias (lower is more uniform)
    bias_w: Dict[str, float] = field(default_factory=dict)
    
    # Bias-P: per-image bias for multi-face images
    bias_p: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    # ENS: Effective Number of Species (higher is more diverse)
    ens: Dict[str, float] = field(default_factory=dict)
    
    # KL Divergence from reference (lower is closer to reference)
    kl_divergence: Dict[str, float] = field(default_factory=dict)
    
    # Distribution counts
    distributions: Dict[str, pd.Series] = field(default_factory=dict)
    
    def summary(self) -> pd.DataFrame:
        """Get summary DataFrame of all metrics."""
        records = []
        for attr_group in ATTRIBUTE_COMBINATIONS:
            attr_name = '_'.join(attr_group)
            records.append({
                'attribute': attr_name,
                'bias_w': self.bias_w.get(attr_name, np.nan),
                'ens': self.ens.get(attr_name, np.nan),
                'kl_divergence': self.kl_divergence.get(attr_name, np.nan),
            })
        return pd.DataFrame(records)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'bias_w': self.bias_w,
            'ens': self.ens,
            'kl_divergence': self.kl_divergence,
            'bias_p_mean': self.bias_p.groupby('Attribute_Group')['Bias-P'].mean().to_dict() 
                          if len(self.bias_p) > 0 else {},
        }


def _get_all_combinations(attribute_group: List[str]) -> List:
    """
    Get all possible value combinations for attribute group.
    
    Args:
        attribute_group: List of attribute names (e.g., ['race', 'gender'])
        
    Returns:
        List of all possible value combinations
    """
    if len(attribute_group) == 1:
        return ATTRIBUTES_DICT[attribute_group[0]]
    else:
        label_universes = [ATTRIBUTES_DICT[attr] for attr in attribute_group]
        return list(product(*label_universes))


def _get_num_categories(attribute_group: List[str]) -> int:
    """Get total number of categories for attribute group."""
    na = 1
    for attr in attribute_group:
        na *= len(ATTRIBUTES_DICT[attr])
    return na


def calculate_bias_w(
    df: pd.DataFrame,
    attribute_groups: Optional[List[List[str]]] = None,
) -> Dict[str, float]:
    """
    Calculate Bias-W (population-level bias) for attribute groups.
    
    Bias-W measures deviation from uniform distribution across all categories.
    Formula: sqrt((1/N_a) * sum_a (freq_a - 1/N_a)^2)
    
    Lower values indicate more uniform (less biased) distribution.
    
    Args:
        df: DataFrame with demographic predictions (columns: race, gender, age)
        attribute_groups: List of attribute groups to analyze.
                         Defaults to ATTRIBUTE_COMBINATIONS.
                         
    Returns:
        Dictionary mapping attribute group name to Bias-W value
    """
    if attribute_groups is None:
        attribute_groups = ATTRIBUTE_COMBINATIONS
    
    results = {}
    
    for attr_group in attribute_groups:
        attr_name = '_'.join(attr_group)
        na = _get_num_categories(attr_group)
        all_combs = _get_all_combinations(attr_group)
        
        # Calculate frequency distribution
        if len(attr_group) == 1:
            freq = df[attr_group[0]].value_counts(normalize=True)
        else:
            freq = df.groupby(attr_group).size() / len(df)
        
        # Calculate Bias-W
        bias_sum = 0.0
        for comb in all_combs:
            if isinstance(comb, str):
                freq_a = freq.get(comb, 0)
            else:
                try:
                    freq_a = freq.loc[comb]
                except KeyError:
                    freq_a = 0
            bias_sum += (freq_a - (1 / na)) ** 2
        
        bias_w = np.sqrt((1 / na) * bias_sum)
        results[attr_name] = float(bias_w)
    
    return results


def calculate_bias_p(
    df: pd.DataFrame,
    image_column: str = 'original_image',
    attribute_groups: Optional[List[List[str]]] = None,
) -> pd.DataFrame:
    """
    Calculate Bias-P (per-image bias) for multi-face images.
    
    Bias-P measures within-image demographic uniformity.
    Only calculated for images with multiple detected faces.
    
    Args:
        df: DataFrame with demographic predictions
        image_column: Column identifying source image
        attribute_groups: List of attribute groups to analyze
        
    Returns:
        DataFrame with columns: Original_Image, Attribute_Group, Bias-P
    """
    if attribute_groups is None:
        attribute_groups = ATTRIBUTE_COMBINATIONS
    
    # Extract original image name if not present
    if image_column not in df.columns:
        if 'face_name_align' in df.columns:
            df = df.copy()
            df[image_column] = df['face_name_align'].apply(
                lambda x: os.path.basename(x).split('_face')[0]
            )
        else:
            raise ValueError(f"Column {image_column} not found in DataFrame")
    
    records = []
    
    for attr_group in attribute_groups:
        attr_name = '_'.join(attr_group)
        na = _get_num_categories(attr_group)
        all_combs = _get_all_combinations(attr_group)
        
        # Calculate per-image bias
        for image_name, group in df.groupby(image_column):
            # Only calculate for multi-face images
            if len(group) <= 1:
                continue
            
            # Calculate frequency within image
            if len(attr_group) == 1:
                freq = group[attr_group[0]].value_counts(normalize=True)
            else:
                freq = group.groupby(attr_group).size() / len(group)
            
            # Calculate Bias-P
            bias_sum = 0.0
            for comb in all_combs:
                if isinstance(comb, str):
                    freq_a = freq.get(comb, 0)
                else:
                    try:
                        freq_a = freq.loc[comb]
                    except KeyError:
                        freq_a = 0
                bias_sum += (freq_a - (1 / na)) ** 2
            
            bias_p = np.sqrt((1 / na) * bias_sum)
            
            records.append({
                'Original_Image': image_name,
                'Attribute_Group': attr_name,
                'Bias-P': float(bias_p),
            })
    
    return pd.DataFrame(records)


def calculate_ens(
    df: pd.DataFrame,
    attribute_groups: Optional[List[List[str]]] = None,
) -> Dict[str, float]:
    """
    Calculate Effective Number of Species (ENS) for attribute groups.
    
    ENS = exp(-sum_g p_g * ln(p_g))
    
    This is the exponential of Shannon entropy. Higher values indicate
    more diversity. Maximum ENS equals the number of categories.
    
    Args:
        df: DataFrame with demographic predictions
        attribute_groups: List of attribute groups to analyze
        
    Returns:
        Dictionary mapping attribute group name to ENS value
    """
    if attribute_groups is None:
        attribute_groups = ATTRIBUTE_COMBINATIONS
    
    results = {}
    
    for attr_group in attribute_groups:
        attr_name = '_'.join(attr_group)
        all_combs = _get_all_combinations(attr_group)
        
        # Filter to valid rows
        mask = df[list(attr_group)].notna().all(axis=1)
        sub = df.loc[mask, list(attr_group)]
        
        if len(sub) == 0:
            results[attr_name] = 0.0
            continue
        
        # Calculate frequency distribution
        if len(attr_group) == 1:
            counts = sub[attr_group[0]].value_counts().to_dict()
            p_list = [counts.get(lab, 0) / len(sub) for lab in ATTRIBUTES_DICT[attr_group[0]]]
        else:
            counts = sub.value_counts().to_dict()
            p_list = [counts.get(tuple(grp) if len(grp) > 1 else grp[0], 0) / len(sub) 
                     for grp in all_combs]
        
        # Calculate ENS = exp(-sum p_g * ln(p_g))
        sum_p_ln_p = 0.0
        for p_g in p_list:
            if p_g > 0:
                sum_p_ln_p += p_g * np.log(p_g)
        
        ens_value = float(np.exp(-sum_p_ln_p))
        results[attr_name] = ens_value
    
    return results


def calculate_kl_divergence(
    df: pd.DataFrame,
    attribute_groups: Optional[List[List[str]]] = None,
    reference_distribution: Optional[Dict[str, Dict]] = None,
    epsilon: float = 1e-12,
) -> Dict[str, float]:
    """
    Calculate KL Divergence from reference distribution.
    
    KL(P||Q) = sum_x P(x) * log(P(x) / Q(x))
    
    where P is the generated distribution and Q is the reference.
    Lower values indicate closer match to reference.
    
    Args:
        df: DataFrame with demographic predictions
        attribute_groups: List of attribute groups to analyze
        reference_distribution: Dict mapping attribute group name to 
                               dict of label->probability.
                               If None, uniform distribution is used.
        epsilon: Small value to avoid log(0)
        
    Returns:
        Dictionary mapping attribute group name to KL divergence value
    """
    if attribute_groups is None:
        attribute_groups = ATTRIBUTE_COMBINATIONS
    
    results = {}
    
    for attr_group in attribute_groups:
        attr_name = '_'.join(attr_group)
        all_combs = _get_all_combinations(attr_group)
        
        # Calculate generated distribution
        if len(attr_group) == 1:
            gen_dist = df[attr_group[0]].value_counts(normalize=True)
        else:
            gen_dist = df.groupby(attr_group).size() / len(df)
        
        # Get reference distribution
        if reference_distribution and attr_name in reference_distribution:
            ref_dist = pd.Series(reference_distribution[attr_name])
        else:
            # Uniform distribution
            ref_dist = pd.Series({label: 1/len(all_combs) for label in all_combs})
        
        # Align distributions
        all_labels = sorted(set(gen_dist.index) | set(ref_dist.index))
        gen_probs = np.array([gen_dist.get(label, 0) for label in all_labels], dtype=float)
        ref_probs = np.array([ref_dist.get(label, 0) for label in all_labels], dtype=float)
        
        # Clip to avoid log(0)
        gen_probs = np.clip(gen_probs, epsilon, 1)
        ref_probs = np.clip(ref_probs, epsilon, 1)
        
        # Renormalize after clipping
        gen_probs = gen_probs / gen_probs.sum()
        ref_probs = ref_probs / ref_probs.sum()
        
        # Calculate KL divergence
        kl_value = float(np.sum(gen_probs * np.log(gen_probs / ref_probs)))
        results[attr_name] = kl_value
    
    return results


def calculate_all_metrics(
    df: pd.DataFrame,
    attribute_groups: Optional[List[List[str]]] = None,
    reference_distribution: Optional[Dict[str, Dict]] = None,
) -> BiasMetrics:
    """
    Calculate all bias and diversity metrics.
    
    Args:
        df: DataFrame with demographic predictions (columns: race, gender, age)
        attribute_groups: List of attribute groups to analyze
        reference_distribution: Reference distribution for KL divergence
        
    Returns:
        BiasMetrics object containing all computed metrics
    """
    if attribute_groups is None:
        attribute_groups = ATTRIBUTE_COMBINATIONS
    
    # Calculate all metrics
    bias_w = calculate_bias_w(df, attribute_groups)
    bias_p = calculate_bias_p(df, attribute_groups=attribute_groups)
    ens = calculate_ens(df, attribute_groups)
    kl_div = calculate_kl_divergence(df, attribute_groups, reference_distribution)
    
    # Calculate distributions
    distributions = {}
    for attr_group in attribute_groups:
        attr_name = '_'.join(attr_group)
        if len(attr_group) == 1:
            distributions[attr_name] = df[attr_group[0]].value_counts(normalize=True)
        else:
            distributions[attr_name] = df.groupby(attr_group).size() / len(df)
    
    return BiasMetrics(
        bias_w=bias_w,
        bias_p=bias_p,
        ens=ens,
        kl_divergence=kl_div,
        distributions=distributions,
    )


def compare_metrics(
    metrics_a: BiasMetrics,
    metrics_b: BiasMetrics,
    name_a: str = "A",
    name_b: str = "B",
) -> pd.DataFrame:
    """
    Compare metrics between two sets of results.
    
    Args:
        metrics_a: First set of metrics
        metrics_b: Second set of metrics
        name_a: Label for first set
        name_b: Label for second set
        
    Returns:
        DataFrame comparing metrics with columns for each set and difference
    """
    summary_a = metrics_a.summary()
    summary_b = metrics_b.summary()
    
    # Merge on attribute
    comparison = summary_a.merge(
        summary_b,
        on='attribute',
        suffixes=(f'_{name_a}', f'_{name_b}')
    )
    
    # Calculate differences
    for metric in ['bias_w', 'ens', 'kl_divergence']:
        col_a = f'{metric}_{name_a}'
        col_b = f'{metric}_{name_b}'
        comparison[f'{metric}_diff'] = comparison[col_b] - comparison[col_a]
    
    return comparison
