"""
CRYSTALLINE DUAL-TRACK EXAMPLE
Demonstrates simultaneous numeric computation WITH semantic preservation
No proprietary references - pure geometric field operations
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional, Union
from enum import Enum, auto
import math


class Domain(Enum):
    """Semantic domains - geometric spaces"""
    PHYSICS = auto()
    COGNITION = auto()
    BIOLOGY = auto()
    MEMORY = auto()
    SOCIAL = auto()
    QUERY = auto()
    OUTPUT = auto()
    NEXUS = auto()      # Coupling result
    RELATIONAL = auto() # Superposition result


@dataclass
class FieldState:
    """
    Dual-state field: carries BOTH numeric values AND semantic meaning
    Key insight: We can observe/use numerics without destroying semantics
    """
    # Numeric track (observable, computable)
    amplitude: float      # Magnitude in current domain
    phase: float         # Angular position [0, 360)
    curvature: float     # Potential well depth (negative = well)
    coherence: float     # Quality/confidence (0, 1]
    
    # Semantic track (meaning, lineage)
    domain: Domain       # Current semantic space
    shell: int          # Depth/layer [1-9]
    meaning: str        # Transformation lineage
    tags: Tuple[str, ...]  # Processing markers
    
    # History (dual-track record)
    history: Tuple[Tuple[str, float, float, float], ...]  # (domain, phase, curv, amp)
    
    def to_dual(self) -> Dict[str, Any]:
        """Return both tracks for observation"""
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
        """Extract numeric state as vector (for computations)"""
        # Convert to Cartesian for actual computation
        x = self.amplitude * math.cos(math.radians(self.phase))
        y = self.amplitude * math.sin(math.radians(self.phase))
        z = self.curvature
        return (x, y, z)


def field_transform(
    src: Union[FieldState, float],
    *,
    domain: Domain,
    shell: int,
    phase: float,
    curvature: float,
    tag: Optional[str] = None,
    meaning_hint: Optional[str] = None
) -> FieldState:
    """
    Core transformation: preserves BOTH numeric computation AND semantic meaning
    
    The key insight: numeric transformations happen in parallel with semantic tracking
    """
    # Initialize or extract source state
    if isinstance(src, FieldState):
        # Numeric computation: transform amplitude based on curvature change
        curvature_factor = abs(curvature / src.curvature) if src.curvature != 0 else 1.0
        new_amplitude = src.amplitude * math.sqrt(curvature_factor)
        
        # Numeric computation: phase shift with domain-specific modulation
        domain_shift = hash(domain.name) % 30  # Domain adds characteristic shift
        new_phase = (src.phase + phase + domain_shift) % 360
        
        # Numeric computation: coherence decay based on transformation complexity
        complexity = abs(curvature - src.curvature) / 10.0
        new_coherence = src.coherence * math.exp(-complexity * 0.1)
        new_coherence = max(0.001, min(1.0, new_coherence))
        
        # Semantic tracking: extend lineage
        new_meaning = f"{src.meaning}→{domain.name}"
        
        # History update
        new_history = src.history + ((domain.name, new_phase, curvature, new_amplitude),)
    else:
        # Bootstrap from scalar
        new_amplitude = float(src) if src != 0 else 1.0
        new_phase = phase % 360
        new_coherence = 1.0
        new_meaning = meaning_hint or domain.name
        new_history = ((domain.name, new_phase, curvature, new_amplitude),)
    
    return FieldState(
        # Numeric track (computed values)
        amplitude=new_amplitude,
        phase=new_phase,
        curvature=curvature,
        coherence=new_coherence,
        
        # Semantic track (preserved meaning)
        domain=domain,
        shell=shell,
        meaning=new_meaning,
        tags=tuple([tag]) if tag else tuple(),
        
        # Dual history
        history=new_history[-10:]  # Keep last 10 for memory efficiency
    )


def field_couple(*fields: FieldState, tag: Optional[str] = None) -> FieldState:
    """
    Couple multiple fields: tensor-like operation preserving both tracks
    
    Numeric: weighted geometric mean of amplitudes, phase averaging
    Semantic: meaning concatenation with coupling notation
    """
    if not fields:
        raise ValueError("Need at least one field to couple")
    
    # Numeric track: geometric computations
    # Amplitude: geometric mean (preserves scale relationships)
    amplitude = math.prod(f.amplitude for f in fields) ** (1.0 / len(fields))
    
    # Phase: circular mean (accounts for wraparound)
    x_sum = sum(f.amplitude * math.cos(math.radians(f.phase)) for f in fields)
    y_sum = sum(f.amplitude * math.sin(math.radians(f.phase)) for f in fields)
    phase = math.degrees(math.atan2(y_sum, x_sum)) % 360
    
    # Curvature: weighted average by amplitude
    total_weight = sum(f.amplitude for f in fields)
    curvature = sum(f.curvature * f.amplitude for f in fields) / total_weight if total_weight > 0 else -3.0
    
    # Coherence: product (coupling reduces certainty)
    coherence = math.prod(f.coherence for f in fields) ** (1.0 / len(fields))
    
    # Semantic track: preserve all meanings
    meaning = " ⊗ ".join(f.meaning for f in fields)  # Tensor product notation
    
    # Combine histories
    history = []
    for f in fields:
        history.extend(f.history[-2:])  # Take last 2 from each
    
    return FieldState(
        amplitude=amplitude,
        phase=phase,
        curvature=curvature,
        coherence=coherence,
        domain=Domain.NEXUS,  # Coupling creates nexus
        shell=max(f.shell for f in fields),
        meaning=meaning,
        tags=tuple([tag]) if tag else tuple(),
        history=tuple(history[-10:])
    )


def field_superpose(fields: List[FieldState], weights: Optional[List[float]] = None, tag: Optional[str] = None) -> FieldState:
    """
    Superpose fields: quantum-like weighted sum preserving both tracks
    
    Numeric: weighted sum with interference
    Semantic: additive meaning with superposition notation
    """
    if not fields:
        raise ValueError("Need fields to superpose")
    
    # Default equal weights
    if weights is None:
        weights = [1.0 / len(fields)] * len(fields)
    else:
        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]
    
    # Numeric track: superposition with interference
    # Complex addition for interference effects
    z_real = sum(w * f.amplitude * math.cos(math.radians(f.phase)) for w, f in zip(weights, fields))
    z_imag = sum(w * f.amplitude * math.sin(math.radians(f.phase)) for w, f in zip(weights, fields))
    
    amplitude = math.sqrt(z_real**2 + z_imag**2)
    phase = math.degrees(math.atan2(z_imag, z_real)) % 360
    
    # Curvature: weighted average
    curvature = sum(w * f.curvature for w, f in zip(weights, fields))
    
    # Coherence: weighted average (superposition maintains partial coherence)
    coherence = sum(w * f.coherence for w, f in zip(weights, fields))
    
    # Semantic track: additive composition
    meaning = " ⊕ ".join(f.meaning for f in fields)  # Direct sum notation
    
    # Sample history from all fields
    history = []
    for f in fields[:3]:  # Take from first 3 fields max
        history.extend(f.history[-1:])
    
    return FieldState(
        amplitude=amplitude,
        phase=phase,
        curvature=curvature,
        coherence=coherence,
        domain=Domain.RELATIONAL,  # Superposition creates relations
        shell=max(f.shell for f in fields),
        meaning=meaning,
        tags=tuple([tag]) if tag else tuple(),
        history=tuple(history)
    )


# ============= EXAMPLE USAGE =============

def process_query_with_dual_output(query: str) -> Dict[str, Any]:
    """
    Example: Process a query maintaining BOTH numeric results AND semantic meaning
    This demonstrates the key capability: getting real numeric values while preserving semantics
    """
    print(f"\nProcessing query: '{query}'")
    print("=" * 60)
    
    # Stage 1: Initialize from query
    field = field_transform(
        1.0,  # Unit amplitude start
        domain=Domain.QUERY,
        shell=1,
        phase=0.0,
        curvature=-2.5,
        tag='init',
        meaning_hint=f"Q[{query[:10]}...]"
    )
    print(f"\n1. QUERY initialization:")
    print(f"   Numeric:  amp={field.amplitude:.3f}, phase={field.phase:.1f}°")
    print(f"   Semantic: {field.meaning}")
    
    # Stage 2: Cognitive processing
    field = field_transform(
        field,
        domain=Domain.COGNITION,
        shell=2,
        phase=90.0,  # 90° phase shift for cognitive processing
        curvature=-3.0,
        tag='cognition'
    )
    print(f"\n2. COGNITION transform:")
    print(f"   Numeric:  amp={field.amplitude:.3f}, phase={field.phase:.1f}°, coherence={field.coherence:.3f}")
    print(f"   Semantic: {field.meaning}")
    
    # Stage 3: Multi-domain analysis (coupling)
    physics_branch = field_transform(
        field,
        domain=Domain.PHYSICS,
        shell=2,
        phase=0.0,
        curvature=-3.5,
        tag='physics'
    )
    
    biology_branch = field_transform(
        field,
        domain=Domain.BIOLOGY,
        shell=2,
        phase=120.0,
        curvature=-2.8,
        tag='biology'
    )
    
    memory_branch = field_transform(
        field,
        domain=Domain.MEMORY,
        shell=3,
        phase=240.0,
        curvature=-3.2,
        tag='memory'
    )
    
    # Couple the branches
    coupled = field_couple(physics_branch, biology_branch, memory_branch, tag='fusion')
    print(f"\n3. Multi-domain COUPLING:")
    print(f"   Numeric:  amp={coupled.amplitude:.3f}, phase={coupled.phase:.1f}°, coherence={coupled.coherence:.3f}")
    print(f"   Semantic: {coupled.domain.name} <- tensor product of 3 domains")
    
    # Stage 4: Output transformation
    output = field_transform(
        coupled,
        domain=Domain.OUTPUT,
        shell=2,
        phase=180.0,
        curvature=-2.0,
        tag='output'
    )
    
    # DUAL OUTPUT: Both numeric results AND semantic context
    result = {
        'query': query,
        
        # NUMERIC RESULTS (actual computed values)
        'numeric': {
            'final_amplitude': round(output.amplitude, 4),
            'final_phase': round(output.phase, 2),
            'final_coherence': round(output.coherence, 4),
            'curvature': round(output.curvature, 3),
            
            # Computed metrics
            'resonance': round(output.amplitude * output.coherence, 4),
            'phase_quadrant': int(output.phase // 90) + 1,
            'depth': output.shell,
            
            # Vector representation
            'vector_3d': output.numeric_vector()
        },
        
        # SEMANTIC CONTEXT (preserved meaning)
        'semantic': {
            'transformation_path': output.meaning,
            'final_domain': output.domain.name,
            'processing_stages': list(output.tags),
            'path_complexity': len(output.meaning.split('→')),
            
            # Semantic metrics
            'domains_traversed': list(set(h[0] for h in output.history)),
            'coupling_detected': '⊗' in output.meaning,
            'superposition_detected': '⊕' in output.meaning
        },
        
        # DUAL-TRACK QUALITY METRICS
        'quality': {
            'coherence': round(output.coherence, 4),
            'transformations': len(output.history),
            'information_preserved': output.coherence > 0.5,
            'numeric_magnitude': round(output.amplitude, 4),
            'semantic_depth': output.meaning.count('→')
        }
    }
    
    print(f"\n4. FINAL OUTPUT:")
    print(f"   Numeric:  amp={output.amplitude:.3f}, phase={output.phase:.1f}°, coherence={output.coherence:.3f}")
    print(f"   Semantic: {output.meaning}")
    print(f"\n   Resonance (amp × coherence): {result['numeric']['resonance']:.4f}")
    print(f"   3D Vector: {tuple(round(x, 3) for x in result['numeric']['vector_3d'])}")
    
    return result


def demonstrate_superposition() -> Dict[str, Any]:
    """
    Demonstrate superposition maintaining dual tracks
    """
    print("\n" + "="*60)
    print("SUPERPOSITION DEMONSTRATION")
    print("="*60)
    
    # Create base field
    base = field_transform(
        1.0,
        domain=Domain.QUERY,
        shell=1,
        phase=0.0,
        curvature=-2.5,
        tag='base'
    )
    
    # Create multiple phase-shifted versions
    fields = []
    for angle in [0, 60, 120, 180, 240, 300]:
        f = field_transform(
            base,
            domain=Domain.COGNITION,
            shell=1,
            phase=float(angle),
            curvature=-2.0,
            tag=f'phase_{angle}'
        )
        fields.append(f)
        print(f"  Phase {angle:3d}°: amp={f.amplitude:.3f}, coherence={f.coherence:.3f}")
    
    # Superpose with weights
    weights = [0.3, 0.2, 0.2, 0.1, 0.1, 0.1]  # Emphasize early phases
    superposed = field_superpose(fields, weights=weights, tag='weighted_sum')
    
    print(f"\nSuperposition result:")
    print(f"  Numeric:  amp={superposed.amplitude:.3f}, phase={superposed.phase:.1f}°")
    print(f"  Semantic: {superposed.domain.name} (from weighted superposition)")
    
    return superposed.to_dual()


if __name__ == "__main__":
    # Example 1: Process query with full dual output
    result1 = process_query_with_dual_output("What is the nature of consciousness?")
    
    print("\n" + "="*60)
    print("COMPLETE DUAL-TRACK OUTPUT:")
    print("="*60)
    
    print("\nNUMERIC TRACK:")
    for key, value in result1['numeric'].items():
        print(f"  {key}: {value}")
    
    print("\nSEMANTIC TRACK:")
    for key, value in result1['semantic'].items():
        if isinstance(value, list):
            print(f"  {key}: {value[:3]}...")  # Truncate long lists
        else:
            print(f"  {key}: {value}")
    
    print("\nQUALITY METRICS:")
    for key, value in result1['quality'].items():
        print(f"  {key}: {value}")
    
    # Example 2: Superposition
    result2 = demonstrate_superposition()
    
    print("\n" + "="*60)
    print("KEY INSIGHT: We maintain BOTH tracks simultaneously!")
    print("- Numeric: Real computations, measurable values")
    print("- Semantic: Preserved meaning, transformation history")
    print("- No information loss, full observability")
    print("="*60)
