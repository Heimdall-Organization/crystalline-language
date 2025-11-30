# CRYSTALLINE LANGUAGE SPECIFICATION v3.1

**A Formal Restricted Python Subset for Hybrid Numeric-Semantic Computation**

**Status**: Complete Formal Specification  
**Version**: 3.1.0  
**Date**: November 2025

---

## ABSTRACT

Crystalline is a formally-specified restricted subset of Python designed as an intermediate representation for hybrid numeric-semantic field computation. Code generators targeting Crystalline emit conformant Python code that operates on semantic fields while preserving both numeric values AND semantic meaning in parallel. This specification defines the complete syntactic, semantic, and type system rules that constitute valid Crystalline programs.

---

## TABLE OF CONTENTS

1. [Introduction and Scope](#1-introduction-and-scope)
2. [Language Philosophy](#2-language-philosophy)
3. [Lexical Structure](#3-lexical-structure)
4. [Syntax (EBNF)](#4-syntax-ebnf)
5. [Type System](#5-type-system)
6. [Semantic Rules](#6-semantic-rules)
7. [Permitted Operations](#7-permitted-operations)
8. [Forbidden Constructs](#8-forbidden-constructs)
9. [Standard Patterns](#9-standard-patterns)
10. [Runtime Requirements](#10-runtime-requirements)
11. [Validation](#11-validation)
12. [Examples](#12-examples)

---

## 1. INTRODUCTION AND SCOPE

### 1.1 What is Crystalline?

Crystalline is a **restricted subset of Python 3.10+** designed for hybrid numeric-semantic computation. It is:

- **An intermediate representation**: Code generators compile higher-level domain-specific languages into Crystalline
- **A dual-track language**: Maintains numeric values AND semantic meaning in parallel
- **Type-safe**: Strong typing prevents unintended information loss
- **Executable Python**: Valid Crystalline is valid Python that runs directly
- **Observable**: Both numeric results AND semantic lineage can be extracted

Crystalline is analogous to:
- Assembly language (low-level, restricted, target of compilation)
- SIMD instructions (parallel computation tracks)
- Automatic differentiation (dual number systems)

But Crystalline is **human-readable Python** with enforced dual numeric-semantic tracking.

### 1.2 Relationship to Python

```
┌─────────────────────────────────────┐
│   Source Languages                  │
│   (Domain-specific notations)       │
└────────────┬────────────────────────┘
             │ Compilation
             ▼
┌─────────────────────────────────────┐
│   CRYSTALLINE                       │
│   (Restricted Python subset)        │
│   - Dual numeric/semantic tracking  │
│   - Parallel computation paths      │
│   - Calls crystalline_core API      │
└────────────┬────────────────────────┘
             │ Execution
             ▼
┌─────────────────────────────────────┐
│   Python 3.10+ Runtime              │
│   + crystalline_core module         │
└─────────────────────────────────────┘
```

### 1.3 Design Goals

**G1 - Dual Preservation**: Maintain both numeric values AND semantic meaning  
**G2 - Traceability**: All transformations maintain complete lineage  
**G3 - Type Safety**: Prevent invalid operations at language level  
**G4 - Observability**: Both tracks can be inspected at any point  
**G5 - Verifiability**: Programs can be mechanically validated  

---

## 2. LANGUAGE PHILOSOPHY

### 2.1 Core Principles

**Principle 1: Dual-Track Computation**

Crystalline maintains TWO parallel computation tracks:

```python
# VALID CRYSTALLINE - Both tracks preserved
result = field_transform(
    input_field,
    domain=Domain.COGNITION,
    shell=2,
    phase=90.0,      # Numeric track
    curvature=-3.0,  # Numeric track
    tag='process'    # Semantic track
)
# result.amplitude: 1.234    (numeric available)
# result.meaning: "QUERY→COGNITION"  (semantic available)
```

**Principle 2: Controlled Numeric Access**

Numeric values can be OBSERVED but not COLLAPSED:

```python
# VALID CRYSTALLINE - Observation without collapse
numeric_value = field.amplitude        # ✓ Read numeric
semantic_trace = field.meaning         # ✓ Read semantic
combined = field.to_dual()             # ✓ Get both

# INVALID CRYSTALLINE - Destructive operations
value = field.amplitude * 2.0          # ❌ Modifies numeric track only
field.phase = 180.0                    # ❌ Mutation forbidden
```

**Principle 3: Immutability with Dual State**

FieldState objects are immutable but carry dual state:

```python
# VALID CRYSTALLINE
field1 = field_transform(...)          # Creates dual-state field
numeric1 = field1.amplitude             # Observe numeric
semantic1 = field1.meaning              # Observe semantic
field2 = field_transform(field1, ...)  # Transform preserves both
```

**Principle 4: Explicit Dual Lineage**

Every field carries BOTH numeric history AND semantic lineage:

```python
field.meaning        # "QUERY→COGNITION→PHYSICS"
field.history        # [(domain, phase, curvature, amplitude), ...]
field.coherence      # 0.831 (numeric quality metric)
```

**Principle 5: Dual Output at Boundaries**

System boundaries can emit BOTH numeric results AND semantic context:

```python
# VALID CRYSTALLINE - Dual output at boundary
return {
    'numeric': {
        'value': final_field.amplitude,
        'phase': final_field.phase,
        'coherence': final_field.coherence
    },
    'semantic': {
        'meaning': final_field.meaning,
        'domain': final_field.domain.name,
        'lineage': final_field.history
    }
}
```

### 2.2 Computational Model

Crystalline views computation as **parallel numeric and semantic transformations**:

- **Numeric Track**: Continuous values (amplitude, phase, curvature)
- **Semantic Track**: Discrete domains and transformation history
- **Coupling**: Integrates multiple fields (preserves both tracks)
- **Superposition**: Weighted combination (numeric) with meaning composition (semantic)
- **Manifolds**: Hierarchical structures maintaining both aspects

---

## 3. LEXICAL STRUCTURE

### 3.1 Tokens

**Reserved Keywords** (Python standard):
```
def, return, if, elif, else, for, while, in, range,
True, False, None, import, from, class, lambda
```

**Crystalline-Specific Identifiers**:
```
Domain, FieldState, DualField, Manifold, ManifoldNode, ArchetypeMeta,
field_transform, field_couple, field_superpose, collapse_field,
observe_numeric, observe_semantic, to_dual,
select_archetype, navigate, select_ending
```

**Literals**:
```python
# Numeric literals (for both parameters AND observation)
1.0, -2.5, 90.0, 360, 0.831

# String literals (for tags, meanings, roles)
'seed', "foundation", '''multi-line'''

# Boolean literals
True, False

# None
None
```

### 3.2 Dual-State Access

```python
# Reading numeric state (VALID)
amplitude = field.amplitude
phase = field.phase
coherence = field.coherence

# Reading semantic state (VALID)
domain = field.domain
meaning = field.meaning
tags = field.tags

# Combined access (VALID)
dual = field.to_dual()  # Returns both tracks
```

---

## 4. SYNTAX (EBNF)

### 4.1 Extended Grammar for Dual Operations

```ebnf
(* ============================================ *)
(* CRYSTALLINE LANGUAGE GRAMMAR v3.1           *)
(* Hybrid numeric-semantic operations          *)
(* ============================================ *)

(* ----- DUAL-STATE ACCESS ----- *)

dual_access    = field_expr , "." , dual_property ;

dual_property  = "amplitude" | "phase" | "curvature" | "coherence"  (* numeric *)
               | "domain" | "meaning" | "tags" | "history"        (* semantic *)
               | "to_dual" , "(" , ")" ;                          (* both *)

observe_expr   = "observe_numeric" , "(" , field_expr , ")"
               | "observe_semantic" , "(" , field_expr , ")" ;

(* ----- DUAL OUTPUT ----- *)

dual_return    = "return" , "{" ,
                 "'numeric'" , ":" , numeric_dict , "," ,
                 "'semantic'" , ":" , semantic_dict ,
                 "}" ;

numeric_dict   = "{" , 
                 "'value'" , ":" , field_expr , "." , "amplitude" , "," ,
                 "'phase'" , ":" , field_expr , "." , "phase" , "," ,
                 "'coherence'" , ":" , field_expr , "." , "coherence" ,
                 "}" ;

semantic_dict  = "{" ,
                 "'meaning'" , ":" , field_expr , "." , "meaning" , "," ,
                 "'domain'" , ":" , field_expr , "." , "domain" , "." , "name" , "," ,
                 "'lineage'" , ":" , field_expr , "." , "history" ,
                 "}" ;
```

---

## 5. TYPE SYSTEM

### 5.1 Core Types

```python
# Dual-state field type
class FieldState:
    # Numeric properties
    amplitude: float      # [0, ∞)
    phase: float         # [0, 360)
    curvature: float     # [-10, 10]
    coherence: float     # (0, 1]
    
    # Semantic properties
    domain: Domain
    shell: int           # [0, 9]
    meaning: str         # Lineage string
    tags: Tuple[str, ...]
    history: Tuple[Tuple[str, float, float, float], ...]
    
    # Dual access
    def to_dual(self) -> Dict[str, Any]
```

### 5.2 Type Rules

```
[T-DUAL-ACCESS]
Γ ⊢ field : FieldState
-------------------------
Γ ⊢ field.amplitude : float
Γ ⊢ field.meaning : str
Γ ⊢ field.to_dual() : Dict

[T-OBSERVE]
Γ ⊢ field : FieldState
----------------------------------
Γ ⊢ observe_numeric(field) : float
Γ ⊢ observe_semantic(field) : str
```

---

## 6. SEMANTIC RULES

### 6.1 Dual Preservation Rules

**Rule D1: Transform Preservation**
```python
# Both tracks flow through transformations
new_field = field_transform(old_field, ...)
assert new_field.amplitude != 0  # Numeric preserved
assert old_field.meaning in new_field.meaning  # Semantic extended
```

**Rule D2: Coupling Preservation**
```python
# Coupling maintains both tracks
coupled = field_couple(field1, field2)
assert coupled.amplitude > 0  # Numeric combined
assert field1.meaning in coupled.meaning  # Semantic merged
```

**Rule D3: Observable Duality**
```python
# Both tracks always observable
assert hasattr(field, 'amplitude')  # Numeric accessible
assert hasattr(field, 'meaning')    # Semantic accessible
```

---

## 7. PERMITTED OPERATIONS

### 7.1 Dual-State Operations

```python
# Read numeric properties
value = field.amplitude           # ✓
angle = field.phase               # ✓
quality = field.coherence         # ✓

# Read semantic properties
domain = field.domain             # ✓
trace = field.meaning             # ✓
labels = field.tags               # ✓

# Transform with dual preservation
new_field = field_transform(      # ✓
    field,
    domain=Domain.PHYSICS,
    phase=180.0,              # Numeric parameter
    tag='physics-transform'   # Semantic parameter
)

# Couple with dual merging
tensor = field_couple(f1, f2, f3) # ✓

# Superpose with dual weighting
result = field_superpose(fields)  # ✓

# Dual output
output = {                        # ✓
    'numeric': field.amplitude,
    'semantic': field.meaning
}
```

### 7.2 Numeric Observations

```python
# VALID - Reading for analysis/display
amplitude = field.amplitude
print(f"Current amplitude: {amplitude}")

# VALID - Conditional logic based on numeric
if field.coherence > 0.5:
    result = field_transform(field, ...)

# VALID - Collecting metrics
metrics = {
    'max_amplitude': max(f.amplitude for f in fields),
    'avg_coherence': sum(f.coherence for f in fields) / len(fields)
}
```

---

## 8. FORBIDDEN CONSTRUCTS

### 8.1 Destructive Operations

```python
# FORBIDDEN - Numeric-only manipulation
new_value = field.amplitude * 2.0     # ❌
field.phase += 90.0                    # ❌

# FORBIDDEN - Semantic-only manipulation  
field.meaning = "custom"               # ❌
field.domain = Domain.PHYSICS          # ❌

# FORBIDDEN - Breaking duality
numeric_only = field.amplitude         # Then discarding field ❌
semantic_only = field.meaning          # Then discarding field ❌
```

### 8.2 Direct Construction

```python
# FORBIDDEN - Manual FieldState creation
field = FieldState(                    # ❌
    amplitude=1.0,
    phase=90.0,
    ...
)

# REQUIRED - Use API functions
field = field_transform(...)           # ✓
```

---

## 9. STANDARD PATTERNS

### 9.1 Dual Pipeline Pattern

```python
def dual_pipeline(seed: Union[str, float]) -> Dict[str, Any]:
    """Pipeline maintaining dual state"""
    # Initialize with dual state
    field = field_transform(
        1.0,                      # Numeric seed
        domain=Domain.QUERY,
        shell=1,
        phase=0.0,                # Numeric
        curvature=-2.5,           # Numeric  
        tag='seed',               # Semantic
        meaning_hint=str(seed)    # Semantic
    )
    
    # Transform preserving both
    field = field_transform(
        field,
        domain=Domain.COGNITION,
        phase=90.0,
        curvature=-3.0,
        tag='cognition'
    )
    
    # Return dual output
    return {
        'numeric': {
            'value': field.amplitude,
            'phase': field.phase,
            'coherence': field.coherence
        },
        'semantic': {
            'meaning': field.meaning,
            'domain': field.domain.name,
            'path': ' → '.join(t for t in field.tags)
        }
    }
```

### 9.2 Dual Analysis Pattern

```python
def analyze_dual_state(fields: List[FieldState]) -> Dict[str, Any]:
    """Analyze both numeric and semantic aspects"""
    # Numeric analysis
    numeric_analysis = {
        'mean_amplitude': sum(f.amplitude for f in fields) / len(fields),
        'phase_distribution': [f.phase for f in fields],
        'coherence_range': (
            min(f.coherence for f in fields),
            max(f.coherence for f in fields)
        )
    }
    
    # Semantic analysis
    domains = set(f.domain for f in fields)
    paths = [f.meaning for f in fields]
    
    semantic_analysis = {
        'unique_domains': [d.name for d in domains],
        'path_complexity': max(len(p.split('→')) for p in paths),
        'common_tags': list(set.intersection(*(set(f.tags) for f in fields)))
    }
    
    return {
        'numeric': numeric_analysis,
        'semantic': semantic_analysis
    }
```

---

## 10. RUNTIME REQUIREMENTS

### 10.1 crystalline_core API

The runtime must provide dual-state operations:

```python
# Core dual-state type
class FieldState:
    # Numeric properties
    amplitude: float
    phase: float
    curvature: float
    coherence: float
    
    # Semantic properties
    domain: Domain
    shell: int
    meaning: str
    tags: Tuple[str, ...]
    history: Tuple[...]
    
    # Dual access
    def to_dual(self) -> Dict[str, Any]:
        return {
            'numeric': {
                'amplitude': self.amplitude,
                'phase': self.phase,
                'curvature': self.curvature,
                'coherence': self.coherence
            },
            'semantic': {
                'domain': self.domain.name,
                'shell': self.shell,
                'meaning': self.meaning,
                'tags': self.tags,
                'history': self.history
            }
        }

# Dual-preserving operations
def field_transform(src, *, domain, shell, phase, curvature, tag=None, meaning_hint=None) -> FieldState:
    """Transform preserving dual state"""
    ...

def observe_numeric(field: FieldState) -> Dict[str, float]:
    """Extract numeric state for observation"""
    ...

def observe_semantic(field: FieldState) -> Dict[str, Any]:
    """Extract semantic state for observation"""
    ...
```

---

## 11. VALIDATION

### 11.1 Dual-State Validation

**DV1: Dual Preservation**
- Every transform maintains both tracks
- No operations destroy either track

**DV2: Observable Duality**
- Both numeric and semantic properties accessible
- to_dual() returns complete state

**DV3: No Track Isolation**
- Operations cannot modify only one track
- Transformations affect both appropriately

### 11.2 Numeric Bounds Validation

```python
assert 0 < field.amplitude < float('inf')
assert 0 <= field.phase < 360
assert -10 <= field.curvature <= 10
assert 0 < field.coherence <= 1
```

### 11.3 Semantic Integrity Validation

```python
assert field.domain in Domain
assert 0 <= field.shell <= 9
assert '→' in field.meaning  # Has lineage
assert isinstance(field.tags, tuple)
```

---

## 12. EXAMPLES

### Example 1: Dual-Output Function

```python
from __future__ import annotations
from typing import Dict, Any, Union

from crystalline_core import (
    Domain,
    FieldState,
    field_transform,
)

def process_with_dual_output(seed: Union[str, float]) -> Dict[str, Any]:
    """Process maintaining both numeric and semantic tracks"""
    # Initialize dual-state field
    field = field_transform(
        1.0,
        domain=Domain.QUERY,
        shell=1,
        phase=0.0,
        curvature=-2.5,
        tag='seed',
        meaning_hint=str(seed)
    )
    
    # Transform through domains
    field = field_transform(
        field,
        domain=Domain.COGNITION,
        shell=2,
        phase=90.0,
        curvature=-3.0,
        tag='analyze'
    )
    
    field = field_transform(
        field,
        domain=Domain.PHYSICS,
        shell=1,
        phase=180.0,
        curvature=-3.5,
        tag='ground'
    )
    
    # Return both numeric results and semantic context
    return {
        'numeric': {
            'amplitude': field.amplitude,
            'phase': field.phase,
            'coherence': field.coherence,
            'curvature': field.curvature
        },
        'semantic': {
            'final_domain': field.domain.name,
            'transformation_path': field.meaning,
            'processing_tags': list(field.tags),
            'depth': field.shell
        },
        'quality': {
            'coherence': field.coherence,
            'transformations': len(field.history)
        }
    }
```

### Example 2: Dual-Track Coupling

```python
from __future__ import annotations
from typing import Dict, Any

from crystalline_core import (
    Domain,
    FieldState,
    field_transform,
    field_couple,
)

def multi_domain_fusion(base: FieldState) -> Dict[str, Any]:
    """Couple multiple domains preserving dual state"""
    # Create domain-specific transforms
    physics = field_transform(
        base,
        domain=Domain.PHYSICS,
        shell=1,
        phase=0.0,
        curvature=-3.0,
        tag='physics'
    )
    
    cognition = field_transform(
        base,
        domain=Domain.COGNITION,
        shell=2,
        phase=120.0,
        curvature=-2.5,
        tag='cognition'
    )
    
    biology = field_transform(
        base,
        domain=Domain.BIOLOGY,
        shell=2,
        phase=240.0,
        curvature=-2.5,
        tag='biology'
    )
    
    # Couple preserving both tracks
    tensor = field_couple(physics, cognition, biology, tag='fusion')
    
    # Extract dual state
    return {
        'numeric': {
            'coupled_amplitude': tensor.amplitude,
            'resultant_phase': tensor.phase,
            'fusion_coherence': tensor.coherence,
            'component_amplitudes': {
                'physics': physics.amplitude,
                'cognition': cognition.amplitude,
                'biology': biology.amplitude
            }
        },
        'semantic': {
            'fusion_domain': tensor.domain.name,
            'combined_meaning': tensor.meaning,
            'component_meanings': {
                'physics': physics.meaning,
                'cognition': cognition.meaning,
                'biology': biology.meaning
            }
        }
    }
```

---

## APPENDIX A: KEY CHANGES FROM v3.0

1. **Dual-State Philosophy**: Instead of forbidding numeric access, we now maintain parallel numeric and semantic tracks

2. **Observable Numerics**: Numeric properties can be read and used for logic/display without destroying semantic meaning

3. **Dual Output**: Functions can return both numeric results AND semantic context

4. **No Proprietary References**: Removed all references to specific proprietary systems

5. **Practical Usability**: Allows the numeric computations needed for real applications while preserving semantic integrity

---

## DOCUMENT METADATA

**Version**: 3.1.0  
**Date**: November 2025  
**Status**: Complete Formal Specification  
**License**: Open Specification

---

END OF SPECIFICATION
