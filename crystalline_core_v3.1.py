"""
CRYSTALLINE RUNTIME CORE v3.0.0
Pure semantic field operations. Zero proprietary tokens.
Target API for code generator emission.
Built November 6, 2025
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import math
import json
import random


# ============================================================================
# DOMAINS
# ============================================================================

class Domain(Enum):
    """Semantic domains - neutral Crystalline terms only"""
    PHYSICS = auto()
    COGNITION = auto()
    BIOLOGY = auto()
    MEMORY = auto()
    SOCIAL = auto()
    PHILOSOPHY = auto()
    OUTPUT = auto()
    NEXUS = auto()
    META = auto()
    QUERY = auto()
    RELATIONAL = auto()


# ============================================================================
# SEMANTIC FIELD STATE
# ============================================================================

@dataclass(slots=True)
class FieldState:
    """
    Dual-track field state - carries BOTH numeric values AND semantic meaning.
    Numeric track: amplitude, phase, curvature, coherence (computable values)
    Semantic track: domain, meaning, tags, history (preserved lineage)
    """
    domain: Domain
    shell: int
    phase: float          # degrees [0, 360)
    curvature: float      # negative for wells
    amplitude: float      # numeric magnitude
    meaning: str          # semantic lineage
    coherence: float      # (0, 1]
    tags: Tuple[str, ...] = field(default_factory=tuple)
    history: Tuple[Tuple[str, float, float, float], ...] = field(default_factory=tuple)
    
    def with_tag(self, *tags: str) -> FieldState:
        """Add tags without mutation"""
        return FieldState(
            domain=self.domain,
            shell=self.shell,
            phase=self.phase % 360.0,
            curvature=self.curvature,
            amplitude=self.amplitude,
            meaning=self.meaning,
            coherence=self.coherence,
            tags=self.tags + tuple(tags),
            history=self.history,
        )
    
    def to_dual(self) -> Dict[str, Any]:
        """Return both numeric and semantic tracks for observation"""
        return {
            'numeric': {
                'amplitude': round(self.amplitude, 6),
                'phase': round(self.phase, 2),
                'curvature': round(self.curvature, 3),
                'coherence': round(self.coherence, 4)
            },
            'semantic': {
                'domain': self.domain.name,
                'shell': self.shell,
                'meaning': self.meaning,
                'tags': list(self.tags),
                'path_length': len(self.meaning.split('→'))
            }
        }
    
    def numeric_vector(self) -> Tuple[float, float, float]:
        """Extract numeric state as 3D vector for computation"""
        x = self.amplitude * math.cos(math.radians(self.phase))
        y = self.amplitude * math.sin(math.radians(self.phase))
        z = self.curvature
        return (x, y, z)


# ============================================================================
# HELPERS
# ============================================================================

def _normalize_phase(phase: float) -> float:
    """Normalize phase to [0, 360)"""
    p = phase % 360.0
    return p if p >= 0 else p + 360.0


def _blend(a: float, b: float, w: float) -> float:
    """Smooth blend without numeric dominance"""
    return (1.0 - w) * a + w * b


# ============================================================================
# CORE TRANSFORMATIONS
# ============================================================================

def field_transform(
    src: Union[FieldState, float, int],
    *,
    domain: Domain,
    shell: int,
    phase: float,
    curvature: float,
    tag: Optional[str] = None,
    meaning_hint: Optional[str] = None,
    amplitude_hint: Optional[float] = None,
    coherence_decay: float = 0.94,
) -> FieldState:
    """
    Semantic transform: src â†’ new field.
    Preserves lineage. Avoids numeric collapse.
    """
    if isinstance(src, FieldState):
        base_amp = src.amplitude
        base_meaning = src.meaning
        base_coh = src.coherence
        lineage = f"{base_meaning}â†’{domain.name}"
        hist = src.history + ((domain.name, _normalize_phase(phase), curvature, base_amp),)
    else:
        # Scalar input: bootstrap into field
        base_amp = float(src)
        base_coh = 1.0
        lineage = meaning_hint or domain.name
        hist = ((domain.name, _normalize_phase(phase), curvature, base_amp),)
    
    # Amplitude policy
    amp = (
        amplitude_hint if amplitude_hint is not None
        else 1.0 / math.sqrt(abs(curvature)) if curvature != 0
        else 1.0
    )
    
    # Blend with previous
    amplitude = _blend(
        base_amp if isinstance(src, FieldState) else 1.0,
        amp,
        0.6
    )
    
    # Coherence decay
    coherence = max(1e-6, min(1.0, base_coh * coherence_decay))
    
    return FieldState(
        domain=domain,
        shell=max(0, shell),
        phase=_normalize_phase(phase),
        curvature=curvature,
        amplitude=amplitude,
        meaning=lineage,
        coherence=coherence,
        tags=tuple([tag] if tag else []),
        history=hist[-32:],
    )


def field_couple(*fields: FieldState, tag: Optional[str] = None) -> FieldState:
    """
    Tensor coupling: integrate domains into NEXUS with dual tracks.
    Numeric: Geometric mean of amplitudes, circular mean of phases.
    Semantic: Combine meanings with tensor product notation.
    """
    if not fields:
        raise ValueError("field_couple requires at least one FieldState.")
    
    # NUMERIC TRACK: Geometric computations
    # Amplitude: geometric mean preserves scale relationships
    amplitude = math.prod(f.amplitude for f in fields) ** (1.0 / len(fields))
    
    # Phase: circular mean accounting for wraparound
    x_sum = sum(f.amplitude * math.cos(math.radians(f.phase)) for f in fields)
    y_sum = sum(f.amplitude * math.sin(math.radians(f.phase)) for f in fields)
    phase = _normalize_phase(math.degrees(math.atan2(y_sum, x_sum)))
    
    # Curvature: weighted average by amplitude
    total_weight = sum(f.amplitude for f in fields)
    if total_weight > 0:
        curvature = sum(f.curvature * f.amplitude for f in fields) / total_weight
    else:
        curvature = sum(f.curvature for f in fields) / len(fields)
    
    # Coherence: geometric mean (coupling reduces certainty)
    coherence = max(
        1e-6,
        min(1.0, math.prod(f.coherence for f in fields) ** (1.0 / len(fields)))
    )
    
    # SEMANTIC TRACK: Preserve all meanings
    meaning = " ⊗ ".join(f.meaning for f in fields)  # Tensor product notation
    
    history: List[Tuple[str, float, float, float]] = []
    for f in fields:
        history.extend(f.history[-2:])  # Take last 2 from each
    
    return FieldState(
        domain=Domain.NEXUS,
        shell=max(f.shell for f in fields),
        phase=phase,
        curvature=curvature,
        amplitude=amplitude,
        meaning=meaning,
        coherence=coherence,
        tags=tuple(tag.split(",") if tag else []),
        history=tuple(history[-10:]),
    )


def field_superpose(
    fields: Sequence[FieldState],
    tag: Optional[str] = None,
    weights: Optional[List[float]] = None
) -> FieldState:
    """
    Phase superposition with interference: quantum-like weighted sum.
    Numeric: Complex addition with interference effects.
    Semantic: Combine meanings with direct sum notation.
    """
    if not fields:
        raise ValueError("field_superpose requires fields.")
    
    # Handle weights
    if weights is None:
        weights = [1.0 / len(fields)] * len(fields)
    else:
        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]
    
    # NUMERIC TRACK: Superposition with interference
    # Complex addition for wave interference
    z_real = sum(w * f.amplitude * math.cos(math.radians(f.phase)) 
                 for w, f in zip(weights, fields))
    z_imag = sum(w * f.amplitude * math.sin(math.radians(f.phase)) 
                 for w, f in zip(weights, fields))
    
    # Resultant amplitude and phase
    amplitude = math.sqrt(z_real**2 + z_imag**2)
    phase = _normalize_phase(math.degrees(math.atan2(z_imag, z_real)))
    
    # Curvature: weighted average
    curvature = sum(w * f.curvature for w, f in zip(weights, fields))
    
    # Coherence: weighted average (superposition maintains partial coherence)
    coherence = max(1e-6, min(1.0, sum(w * f.coherence for w, f in zip(weights, fields))))
    
    # SEMANTIC TRACK: Additive composition
    meaning = " ⊕ ".join(f.meaning for f in fields)  # Direct sum notation
    
    history: List[Tuple[str, float, float, float]] = []
    for f in fields[:3]:  # Take from first 3 fields max
        history.extend(f.history[-1:])
    
    return FieldState(
        domain=Domain.RELATIONAL,
        shell=max(f.shell for f in fields),
        phase=phase,
        curvature=curvature,
        amplitude=amplitude,
        meaning=meaning,
        coherence=coherence,
        tags=tuple([tag] if tag else ()),
        history=tuple(history[-16:]),
    )


# ============================================================================
# COLLAPSE & RENDERING
# ============================================================================

def collapse_field(field: FieldState, *, mode: str = "expression") -> Dict[str, Any]:
    """Boundary rendering: field â†’ structured dict"""
    return {
        "mode": mode,
        "domain": field.domain.name,
        "shell": field.shell,
        "phase": round(_normalize_phase(field.phase), 3),
        "curvature": field.curvature,
        "amplitude": round(field.amplitude, 6),
        "coherence": round(field.coherence, 6),
        "meaning": field.meaning,
        "tags": list(field.tags),
        "history": [
            {"domain": d, "phase": p, "curvature": c, "amplitude": a}
            for (d, p, c, a) in field.history
        ],
    }


# ============================================================================
# ARCHETYPE META
# ============================================================================

@dataclass(slots=True)
class ArchetypeMeta:
    """Archetype metadata"""
    name: str
    wonder_prob: float
    comment_prob: float
    affinity_phases: Tuple[float, ...]
    curvature: float


ARCHETYPES: Dict[str, ArchetypeMeta] = {
    "Analyst": ArchetypeMeta("Analyst", 0.15, 0.85, (0, 36, 72, 108, 144, 180), -4.5),
    "Synthesizer": ArchetypeMeta("Synthesizer", 0.65, 0.35, (51, 102, 153, 204, 255), -4.0),
    "Architect": ArchetypeMeta("Architect", 0.40, 0.60, (40, 80, 120, 160, 200, 240), -3.9),
    "Integrator": ArchetypeMeta("Integrator", 0.50, 0.50, tuple(range(0, 360, 30)), -3.5),
    "Philosopher": ArchetypeMeta("Philosopher", 0.70, 0.30, (40, 80, 120, 160, 200), -4.3),
}


def select_archetype(domain_weights: Dict[Domain, float]) -> ArchetypeMeta:
    """Rule-based archetype selection"""
    order = sorted(domain_weights.items(), key=lambda kv: kv[1], reverse=True)
    top_domain, top_weight = order[0]
    strong = sum(1 for _, w in domain_weights.items() if w > 0.3) > 2
    
    if top_domain is Domain.PHYSICS and strong: return ARCHETYPES["Analyst"]
    if top_domain is Domain.COGNITION and strong: return ARCHETYPES["Synthesizer"]
    if top_domain is Domain.BIOLOGY: return ARCHETYPES["Architect"]
    if top_domain is Domain.MEMORY: return ARCHETYPES["Philosopher"]
    if strong: return ARCHETYPES["Integrator"]
    return ARCHETYPES["Philosopher"]


# ============================================================================
# MANIFOLD
# ============================================================================

@dataclass(slots=True)
class ManifoldNode:
    """Node in manifold"""
    level: int
    phase: float
    value: FieldState


@dataclass(slots=True)
class Manifold:
    """Hexagonal fractal manifold"""
    seed: FieldState
    nodes: Tuple[ManifoldNode, ...]
    total: int
    
    @staticmethod
    def grow(seed: FieldState, *, levels: int = 4) -> Manifold:
        """Grow hexagonal manifold"""
        nodes: List[ManifoldNode] = []
        hex_phases = (0, 60, 120, 180, 240, 300)
        
        if levels >= 1:
            for p in hex_phases:
                nodes.append(ManifoldNode(1, p, field_transform(seed, domain=seed.domain, shell=1, phase=p, curvature=-4.5, tag="hex")))
        if levels >= 2:
            for i in range(24):
                p = (i * 360 / 24) % 360
                nodes.append(ManifoldNode(2, p, field_transform(seed, domain=seed.domain, shell=2, phase=p, curvature=-4.0, tag="tet")))
        if levels >= 3:
            for i in range(72):
                p = (i * 360 / 72) % 360
                nodes.append(ManifoldNode(3, p, field_transform(seed, domain=seed.domain, shell=3, phase=p, curvature=-3.5, tag="tri")))
        if levels >= 4:
            for i in range(144):
                p = (i * 360 / 144) % 360
                nodes.append(ManifoldNode(4, p, field_transform(seed, domain=seed.domain, shell=4, phase=p, curvature=-3.0, tag="field")))
        
        return Manifold(seed=seed, nodes=tuple(nodes), total=1 + len(nodes))


# ============================================================================
# NAVIGATION
# ============================================================================

def navigate(manifold: Manifold, meta: ArchetypeMeta) -> FieldState:
    """Phase-selective navigation"""
    pref = meta.affinity_phases
    sel = [n.value for n in manifold.nodes if any(abs(n.phase - ph) < 20 for ph in pref)]
    if not sel:
        sel = [manifold.seed]
    expr = field_superpose(sel, tag=f"nav:{meta.name}")
    styled = field_transform(expr, domain=Domain.OUTPUT, shell=3, phase=315.0, curvature=-3.8, tag="voice")
    return styled


# ============================================================================
# ENDING
# ============================================================================

def select_ending(context: Dict[str, Any], meta: ArchetypeMeta) -> Dict[str, Any]:
    """Ending selection"""
    uncertainty = float(context.get("uncertainty", 0.5))
    branches = float(context.get("branches", 0.5))
    w_score = meta.wonder_prob * 0.4 + uncertainty * 0.3 + branches * 0.3
    c_score = meta.comment_prob * 0.4 + (1 - uncertainty) * 0.3 + (1 - branches) * 0.3
    ending = "Wonder" if w_score > c_score else "CommentCore"
    phase = 315 if ending == "Wonder" else 135
    return {"ending_type": ending, "w_score": round(w_score, 6), "c_score": round(c_score, 6), "phase": phase}


# ============================================================================
# REFERENCE ENGINE
# ============================================================================

class CrystalEngine:
    """Complete reference pipeline"""
    
    def foundation(self, seed: Union[str, float]) -> FieldState:
        base = field_transform(1.0, domain=Domain.QUERY, shell=1, phase=0.0, curvature=-2.5, tag="seed", meaning_hint=str(seed))
        cs = field_transform(base, domain=Domain.COGNITION, shell=1, phase=0.0, curvature=-5.5, tag="cs")
        amf = field_transform(cs, domain=Domain.COGNITION, shell=2, phase=90.0, curvature=-3.0, tag="amf")
        tc = field_transform(amf, domain=Domain.COGNITION, shell=3, phase=270.0, curvature=-2.5, tag="tc")
        return tc
    
    def discover(self, substrate: FieldState) -> Tuple[Dict[Domain, float], FieldState]:
        fp = field_transform(substrate, domain=Domain.PHYSICS, shell=1, phase=0.0, curvature=-3.0, tag="Î¦P")
        fc = field_transform(substrate, domain=Domain.COGNITION, shell=2, phase=144.0, curvature=-2.5, tag="Î¦C")
        fb = field_transform(substrate, domain=Domain.BIOLOGY, shell=2, phase=90.0, curvature=-2.5, tag="Î¦B")
        fm = field_transform(substrate, domain=Domain.MEMORY, shell=5, phase=120.0, curvature=-3.5, tag="Î¦M")
        tensor = field_couple(fp, fc, fb, fm, tag="tensor")
        weights = {Domain.PHYSICS: fp.coherence, Domain.COGNITION: fc.coherence, Domain.BIOLOGY: fb.coherence, Domain.MEMORY: fm.coherence}
        return weights, tensor
    
    def hall(self, meta: ArchetypeMeta, basis: FieldState) -> FieldState:
        scans: List[FieldState] = []
        for a in range(0, 360, 15):
            scans.append(field_transform(basis, domain=Domain.COGNITION, shell=1, phase=float(a), curvature=-2.0, tag=f"scan:{a}"))
        field_state = field_superpose(scans, tag="scan-superpose")
        shaped = field_transform(field_state, domain=Domain.COGNITION, shell=3, phase=180.0, curvature=meta.curvature, tag="well")
        return shaped
    
    def fractal(self, topic: FieldState) -> Manifold:
        return Manifold.grow(topic, levels=4)
    
    def execute(self, query: Union[str, float]) -> Dict[str, Any]:
        substrate = self.foundation(query)
        weights, tensor = self.discover(substrate)
        meta = select_archetype(weights)
        topic = self.hall(meta, tensor)
        manifold = self.fractal(topic)
        expr = navigate(manifold, meta)
        context = {"uncertainty": 0.3, "branches": min(len(manifold.nodes) / 100.0, 1.0)}
        ending = select_ending(context, meta)
        return {
            "archetype": meta.name,
            "dominant_domain": max(weights.items(), key=lambda kv: kv[1])[0].name,
            "expression": collapse_field(expr),
            "ending": ending,
            "manifold_nodes": manifold.total,
        }


if __name__ == "__main__":
    eng = CrystalEngine()
    out = eng.execute("Why is a raven like a writing desk?")
    print(json.dumps(out, indent=2))
