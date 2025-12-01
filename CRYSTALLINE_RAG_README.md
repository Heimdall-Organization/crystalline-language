# Crystalline RAG - Physics-Guided Retrieval-Augmented Generation

A novel RAG (Retrieval-Augmented Generation) system built on Crystalline's dual-track computation model, combining physics-guided semantic search with explainable retrieval.

## Overview

Traditional RAG systems use vector embeddings that are "black boxes" - you get similarity scores but lose semantic meaning. **Crystalline RAG** maintains BOTH numeric similarity AND semantic lineage throughout the entire retrieval pipeline.

### Key Innovation

Every document embedding is a `FieldState` object that carries:
- **Numeric Track**: amplitude, phase, curvature (for similarity computation)
- **Semantic Track**: domain, meaning, transformation history (for explainability)

This dual-track approach enables:
- ✓ **Explainable retrieval** - trace exactly why documents were matched
- ✓ **Physics-guided similarity** - use field operations (coupling, superposition) for matching
- ✓ **Golden angle spacing** - optimal document organization using φ = 137.5°
- ✓ **Coherent synthesis** - combine multiple documents via field superposition

## Architecture

```
┌─────────────┐
│  Document   │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│  Text Processing (Chunking)                 │
│  • Golden angle phase spacing               │
│  • Overlap-based chunking                   │
└──────┬──────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│  Crystalline Embedding Generation           │
│  QUERY → COGNITION → PHYSICS → MEMORY       │
│  • Extract text features                    │
│  • Multi-domain field transforms            │
│  • Final coupling into NEXUS domain         │
└──────┬──────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│  Vector Store (Field-based)                 │
│  • Stores (chunk, FieldState) pairs         │
│  • Maintains full dual-track information    │
└──────┬──────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│  Query Processing                           │
│  • Transform query into FieldState          │
│  • Compute dual-track similarity            │
│  • Phase alignment + amplitude correlation  │
└──────┬──────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│  Response Generation                        │
│  • Field superposition of top chunks        │
│  • Weighted by similarity scores            │
│  • Transform to OUTPUT domain               │
└─────────────────────────────────────────────┘
```

## Usage

### Basic Example

```python
from crystalline_rag import CrystallineRAG

# Initialize RAG system
rag = CrystallineRAG(chunk_size=512, chunk_overlap=64)

# Add documents
rag.add_document(
    doc_id="doc1",
    content="Your document text here...",
    metadata={"source": "manual", "topic": "physics"}
)

# Query and generate response
response = rag.query_and_generate(
    query_text="What is quantum field theory?",
    top_k=5,
    max_context_chunks=3
)

# Examine results
print(f"Query: {response['query']}")
print(f"Synthesis: {response['answer']['synthesis']}")
print(f"Coherence: {response['field_analysis']['coherence']}")
```

### Advanced: Direct Retrieval

```python
# Just retrieve, don't generate
results = rag.query(
    query_text="machine learning",
    top_k=10,
    use_interference=True  # Use quantum-like interference scoring
)

for result in results:
    print(f"Chunk: {result.chunk.content[:100]}")
    print(f"Score: {result.numeric_score}")
    print(f"Phase Alignment: {result.phase_alignment}")
    print(f"Semantic Lineage: {result.semantic_lineage}")
    print(f"Coherence: {result.coherence}")
```

## Components

### 1. Text Processing (`CrystallineTextProcessor`)

Chunks documents using:
- Word-based chunking with configurable size/overlap
- Golden angle phase spacing (φ = 137.5°) for optimal retrieval
- Feature extraction (complexity, sentiment, question detection)

### 2. Embedding Generation (`CrystallineEmbedder`)

Transforms text through multi-domain pipeline:

```python
Text
  ↓ (QUERY domain - initialization)
  ↓ (COGNITION domain - semantic processing, golden angle phase)
  ↓ (PHYSICS domain - numeric patterns, sentiment-based phase)
  ↓ (MEMORY domain - contextual info, question detection)
  ↓ (field_couple - tensor product into NEXUS)
Embedding (FieldState)
```

Each transform preserves **both** numeric values **and** semantic meaning.

### 3. Vector Store (`CrystallineVectorStore`)

Stores `(DocumentChunk, FieldState)` pairs with:
- Document indexing for fast lookup
- Full dual-track preservation
- No information loss

### 4. Similarity Computation (`CrystallineSimilarity`)

Computes similarity using multiple metrics:

**Standard Similarity:**
```
score = 0.35 × phase_alignment +
        0.25 × amplitude_ratio +
        0.20 × curvature_similarity +
        0.20 × vector_similarity
```

**Interference Similarity** (optional):
- Uses `field_superpose` to create quantum-like interference
- Constructive interference = high similarity
- Destructive interference = low similarity

### 5. Response Generation

Combines top-k chunks via:
1. `field_superpose` with score-based weights
2. Transform to OUTPUT domain
3. Return with full field analysis

## Dual-Track Computation

Every operation maintains both tracks:

### Numeric Track
- `amplitude`: Magnitude of information
- `phase`: Angular position (0-360°)
- `curvature`: Potential well depth
- `coherence`: Quality metric (0-1]

### Semantic Track
- `domain`: Conceptual space (QUERY, COGNITION, PHYSICS, etc.)
- `shell`: Abstraction layer (0-9)
- `meaning`: Transformation lineage ("A→B→C")
- `tags`: Processing markers
- `history`: Complete transformation record

## Similarity Metrics Explained

### Phase Alignment
```python
phase_diff = abs(field1.phase - field2.phase)
if phase_diff > 180:
    phase_diff = 360 - phase_diff
phase_alignment = cos(radians(phase_diff))
```

Range: [-1, 1]
- 1.0 = perfectly aligned (same phase)
- 0.0 = orthogonal (90° apart)
- -1.0 = opposite (180° apart)

### Amplitude Correlation
```python
amp_ratio = min(amp1, amp2) / max(amp1, amp2)
```

Range: [0, 1]
- 1.0 = same magnitude
- 0.0 = vastly different magnitude

### Curvature Similarity
```python
curv_similarity = exp(-abs(curv1 - curv2) / 5.0)
```

Range: (0, 1]
- 1.0 = identical potential wells
- decays exponentially with difference

### Vector Cosine Similarity

3D vectors from (amplitude, phase, curvature):
```python
v = (amp × cos(phase), amp × sin(phase), curvature)
```

Standard cosine similarity on these vectors.

## Examples from Demo

### Query: "What is Crystalline?"

**Results:**
```
Top Match: doc3_physics
Score: 0.1716
Phase Alignment: -0.9784
Coherence: 0.689

Semantic Lineage:
Text[Field theory...]→COGNITION ⊗
Text[Field theory...]→COGNITION→PHYSICS ⊗
Text[Field theory...]→COGNITION→PHYSICS→MEMORY
```

Shows full transformation path through semantic domains!

### Query: "How do fields work?"

**Field Analysis:**
```
output_domain: OUTPUT
coherence: 0.7807
phase: 180.0°
amplitude: 0.6613
```

Combined 2 chunks via superposition with coherence 0.78.

## Performance Characteristics

### Strengths
- **Explainability**: Full semantic lineage preserved
- **Flexibility**: Multiple similarity metrics combined
- **Novelty**: Physics-guided approach is unique
- **Quality**: Coherence metric tracks information quality

### Limitations
- **Computational**: More expensive than simple vector dot products
- **Scale**: Currently in-memory only
- **Embeddings**: Not neural embeddings (different paradigm)

## Comparison to Traditional RAG

| Aspect | Traditional RAG | Crystalline RAG |
|--------|----------------|-----------------|
| Embeddings | Dense vectors (768-1536 dims) | FieldState (dual-track) |
| Similarity | Cosine/dot product | Multi-metric field-based |
| Explainability | Low (black box) | High (full lineage) |
| Synthesis | Concatenation | Field superposition |
| Information | Numeric only | Numeric + semantic |
| Paradigm | Statistical | Physics-guided |

## Advanced Topics

### Golden Angle Spacing

Each chunk gets phase:
```python
phase = (chunk_idx × 137.5077°) % 360°
```

This golden angle (φ) creates optimal spacing that:
- Avoids rational fraction resonances
- Maximizes phase space coverage
- Minimizes destructive interference

### Field Coupling vs Superposition

**Coupling** (`field_couple`):
- Tensor product: A ⊗ B
- Creates NEXUS domain
- Integrates multiple dimensions
- Used for: embedding generation

**Superposition** (`field_superpose`):
- Direct sum: A ⊕ B
- Creates RELATIONAL domain
- Quantum-like interference
- Used for: response synthesis

### Coherence Decay

Coherence decreases with each transformation:
```python
new_coherence = coherence × exp(-complexity × 0.1)
```

This models information loss through processing pipeline.

## Future Enhancements

1. **Persistent Storage**: Add database backend (PostgreSQL, Qdrant)
2. **Hybrid Embeddings**: Combine with neural embeddings
3. **Reranking**: Use Crystalline as reranker for neural RAG
4. **Multi-modal**: Extend to images, audio via field representations
5. **Query Expansion**: Use field transforms to expand queries
6. **Feedback Learning**: Adjust curvature/phases based on relevance feedback

## Mathematical Foundation

Based on:
- Variational calculus (energy minimization)
- Electromagnetic field theory (amplitude, phase)
- Geometric optimization (golden angle)
- Quantum mechanics (superposition, interference)

See `CRYSTALLINE_LANGUAGE_SPEC_v3.1.md` for formal specification.

## Requirements

- Python 3.10+
- Dependencies from `crystalline_core_v3.1.py`

## License

Apache 2.0 (same as Crystalline)

## Citation

If you use Crystalline RAG in research:

```bibtex
@software{crystalline_rag,
  title={Crystalline RAG: Physics-Guided Retrieval-Augmented Generation},
  author={Built on Crystalline v3.1},
  year={2025},
  note={Dual-track computation for explainable document retrieval}
}
```

## Contributing

Contributions welcome! Areas of interest:
- Database backends
- Performance optimizations
- Neural embedding integration
- Multi-modal support
- Benchmarking vs traditional RAG

---

**Built with Crystalline v3.1 - Where Physics Meets Code Synthesis**
