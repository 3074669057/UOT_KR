"""Reliability audit for RC-UOT-v2.1."""
from __future__ import annotations
import numpy as np

def audit_q_distribution(q: np.ndarray, name: str = 'q') -> dict:
    """Audit q values for validity."""
    return {
        'name': name,
        'q_min': float(np.min(q)),
        'q_max': float(np.max(q)),
        'q_mean': float(np.mean(q)),
        'q_std': float(np.std(q)),
        'q_unique_count': len(np.unique(np.round(q, 4))),
        'q_nonconstant': bool(np.std(q) > 0.01),
    }