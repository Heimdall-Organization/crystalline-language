"""
CRYSTALLINE RAG (Retrieval-Augmented Generation) v1.0
Physics-guided document retrieval using field-based semantic search
Built on Crystalline v3.1 dual-track computation model

Key Innovation: Document embeddings maintain BOTH semantic meaning AND numeric vectors
throughout the entire retrieval pipeline, enabling explainable, physics-guided search.
"""

import sys
import os

# Add the directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Read and execute the core file to get the needed classes/functions
# Create a fake module and add it to sys.modules so dataclasses work
import types
crystalline_core_module = types.ModuleType('crystalline_core')
sys.modules['crystalline_core'] = crystalline_core_module

# Execute the core code in the module's namespace
with open(os.path.join(os.path.dirname(__file__), 'crystalline_core_v3.1.py')) as f:
    core_code = f.read()
    exec(core_code, crystalline_core_module.__dict__)

# Import the needed symbols into our namespace
Domain = crystalline_core_module.Domain
FieldState = crystalline_core_module.FieldState
field_transform = crystalline_core_module.field_transform
field_couple = crystalline_core_module.field_couple
field_superpose = crystalline_core_module.field_superpose
collapse_field = crystalline_core_module.collapse_field

# The exec above will have loaded these into the global namespace
from dataclasses import dataclass, field as dc_field
from typing import List, Dict, Any, Tuple, Optional, Union
import math
import re
from collections import defaultdict


# ============================================================================
# DOCUMENT REPRESENTATION
# ============================================================================

@dataclass
class Document:
    """Document with dual-track field representation"""
    doc_id: str
    content: str
    metadata: Dict[str, Any]
    field_state: Optional[FieldState] = None
    chunks: List['DocumentChunk'] = dc_field(default_factory=list)


@dataclass
class DocumentChunk:
    """Chunk of a document with its own field representation"""
    chunk_id: str
    doc_id: str
    content: str
    start_pos: int
    end_pos: int
    field_state: Optional[FieldState] = None


@dataclass
class RetrievalResult:
    """Result from retrieval with dual-track scoring"""
    chunk: DocumentChunk
    numeric_score: float  # Numeric similarity
    semantic_lineage: str  # Semantic transformation path
    coherence: float  # Quality metric
    phase_alignment: float  # Phase similarity (0-1)


# ============================================================================
# TEXT PROCESSING WITH FIELD TRANSFORMS
# ============================================================================

class CrystallineTextProcessor:
    """Process text using field transforms to preserve semantic structure"""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.golden_angle = 137.5077  # Golden angle for optimal spacing

    def chunk_document(self, doc: Document) -> List[DocumentChunk]:
        """
        Chunk document with golden angle phase spacing for optimal retrieval
        Each chunk gets a unique phase position based on golden angle
        """
        text = doc.content
        chunks = []

        # Simple word-based chunking
        words = text.split()
        chunk_idx = 0
        start_idx = 0

        while start_idx < len(words):
            end_idx = min(start_idx + self.chunk_size, len(words))
            chunk_text = ' '.join(words[start_idx:end_idx])

            # Calculate character positions
            char_start = len(' '.join(words[:start_idx]))
            char_end = char_start + len(chunk_text)

            chunk = DocumentChunk(
                chunk_id=f"{doc.doc_id}_chunk_{chunk_idx}",
                doc_id=doc.doc_id,
                content=chunk_text,
                start_pos=char_start,
                end_pos=char_end
            )
            chunks.append(chunk)

            # Move to next chunk with overlap
            start_idx += (self.chunk_size - self.chunk_overlap)
            chunk_idx += 1

        return chunks

    def extract_features(self, text: str) -> Dict[str, float]:
        """Extract text features for field initialization"""
        # Simple feature extraction
        word_count = len(text.split())
        char_count = len(text)
        avg_word_len = char_count / max(word_count, 1)

        # Sentiment approximation (simple heuristic)
        positive_words = len(re.findall(r'\b(good|great|excellent|best|amazing)\b', text.lower()))
        negative_words = len(re.findall(r'\b(bad|poor|worst|terrible|awful)\b', text.lower()))
        sentiment = (positive_words - negative_words) / max(word_count, 1)

        # Complexity (longer words = more complex)
        complexity = avg_word_len / 10.0

        # Question detection
        has_question = 1.0 if '?' in text else 0.0

        return {
            'word_count': word_count,
            'complexity': complexity,
            'sentiment': sentiment,
            'has_question': has_question,
            'density': word_count / max(char_count, 1)
        }


# ============================================================================
# CRYSTALLINE EMBEDDING GENERATOR
# ============================================================================

class CrystallineEmbedder:
    """
    Generate embeddings using Crystalline's field transforms

    Key insight: Embeddings are FieldState objects that maintain:
    - Numeric: amplitude, phase, curvature (for similarity computation)
    - Semantic: domain, meaning, lineage (for explainability)
    """

    def __init__(self):
        self.golden_angle = 137.5077

    def embed_text(self, text: str, chunk_idx: int = 0) -> FieldState:
        """
        Transform text into field representation using multi-domain analysis

        Pipeline:
        1. QUERY domain: Initialize from text features
        2. COGNITION domain: Process semantic content
        3. PHYSICS domain: Extract numeric patterns
        4. MEMORY domain: Capture context
        5. NEXUS domain: Couple all dimensions
        """
        processor = CrystallineTextProcessor()
        features = processor.extract_features(text)

        # Stage 1: Initialize in QUERY domain
        # Use word count to set initial amplitude
        init_amplitude = math.log1p(features['word_count'])

        query_field = field_transform(
            init_amplitude,
            domain=Domain.QUERY,
            shell=1,
            phase=0.0,
            curvature=-2.5,
            tag='text_init',
            meaning_hint=f"Text[{text[:20]}...]"
        )

        # Stage 2: COGNITION - semantic processing
        # Phase based on golden angle * chunk index for optimal spacing
        cognition_phase = (chunk_idx * self.golden_angle) % 360

        cognition_field = field_transform(
            query_field,
            domain=Domain.COGNITION,
            shell=2,
            phase=cognition_phase,
            curvature=-3.0 - features['complexity'],  # Complexity affects curvature
            tag='semantic'
        )

        # Stage 3: PHYSICS - numeric patterns
        # Phase reflects sentiment
        physics_phase = 180.0 + (features['sentiment'] * 90.0)  # Range: 90-270°

        physics_field = field_transform(
            cognition_field,
            domain=Domain.PHYSICS,
            shell=2,
            phase=physics_phase,
            curvature=-3.5,
            tag='numeric'
        )

        # Stage 4: MEMORY - contextual information
        # Phase based on whether it's a question
        memory_phase = 315.0 if features['has_question'] > 0.5 else 45.0

        memory_field = field_transform(
            physics_field,
            domain=Domain.MEMORY,
            shell=3,
            phase=memory_phase,
            curvature=-2.8,
            tag='context'
        )

        # Stage 5: NEXUS - couple all dimensions
        embedding = field_couple(
            cognition_field,
            physics_field,
            memory_field,
            tag='embedding'
        )

        return embedding

    def embed_chunk(self, chunk: DocumentChunk, chunk_idx: int) -> FieldState:
        """Embed a document chunk"""
        return self.embed_text(chunk.content, chunk_idx)


# ============================================================================
# VECTOR STORE (Field-based)
# ============================================================================

class CrystallineVectorStore:
    """
    Vector store using FieldState for dual-track storage
    Maintains both numeric vectors AND semantic meaning
    """

    def __init__(self):
        self.embeddings: List[Tuple[DocumentChunk, FieldState]] = []
        self.doc_index: Dict[str, List[int]] = defaultdict(list)  # doc_id -> indices

    def add_chunk(self, chunk: DocumentChunk, embedding: FieldState):
        """Add chunk with its embedding"""
        idx = len(self.embeddings)
        self.embeddings.append((chunk, embedding))
        self.doc_index[chunk.doc_id].append(idx)

    def add_chunks(self, chunks: List[DocumentChunk], embeddings: List[FieldState]):
        """Add multiple chunks"""
        for chunk, emb in zip(chunks, embeddings):
            self.add_chunk(chunk, emb)

    def get_all_chunks(self) -> List[Tuple[DocumentChunk, FieldState]]:
        """Get all stored chunks with embeddings"""
        return self.embeddings

    def get_doc_chunks(self, doc_id: str) -> List[Tuple[DocumentChunk, FieldState]]:
        """Get all chunks for a specific document"""
        indices = self.doc_index.get(doc_id, [])
        return [self.embeddings[i] for i in indices]


# ============================================================================
# SIMILARITY COMPUTATION (Field-based)
# ============================================================================

class CrystallineSimilarity:
    """
    Compute similarity using field operations

    Key innovation: Uses field_couple and phase interference for similarity
    """

    @staticmethod
    def compute_similarity(field1: FieldState, field2: FieldState) -> Dict[str, float]:
        """
        Compute dual-track similarity:
        - Numeric: Phase alignment, amplitude correlation, curvature match
        - Semantic: Domain compatibility, coherence product
        """
        # Phase alignment (cosine of phase difference)
        phase_diff = abs(field1.phase - field2.phase)
        if phase_diff > 180:
            phase_diff = 360 - phase_diff
        phase_alignment = math.cos(math.radians(phase_diff))

        # Amplitude correlation (geometric mean ratio)
        amp_ratio = min(field1.amplitude, field2.amplitude) / max(field1.amplitude, field2.amplitude, 1e-6)

        # Curvature similarity (how similar are the potential wells)
        curv_diff = abs(field1.curvature - field2.curvature)
        curv_similarity = math.exp(-curv_diff / 5.0)  # Decay with difference

        # Coherence product (both must be coherent for good match)
        coherence_product = field1.coherence * field2.coherence

        # Vector cosine similarity
        v1 = field1.numeric_vector()
        v2 = field2.numeric_vector()
        dot_product = sum(a * b for a, b in zip(v1, v2))
        mag1 = math.sqrt(sum(x**2 for x in v1))
        mag2 = math.sqrt(sum(x**2 for x in v2))
        vector_similarity = dot_product / max(mag1 * mag2, 1e-6)

        # Combined numeric score (weighted average)
        numeric_score = (
            0.35 * phase_alignment +
            0.25 * amp_ratio +
            0.20 * curv_similarity +
            0.20 * vector_similarity
        )

        # Apply coherence as quality multiplier
        final_score = numeric_score * coherence_product

        return {
            'numeric_score': final_score,
            'phase_alignment': phase_alignment,
            'amplitude_ratio': amp_ratio,
            'curvature_similarity': curv_similarity,
            'vector_similarity': vector_similarity,
            'coherence': coherence_product
        }

    @staticmethod
    def compute_interference_score(query_field: FieldState, doc_field: FieldState) -> float:
        """
        Use superposition to compute interference-based similarity
        Constructive interference = high similarity
        """
        # Superpose query and document fields
        superposed = field_superpose([query_field, doc_field], weights=[0.5, 0.5])

        # Strong constructive interference = high amplitude
        # Destructive interference = low amplitude
        max_possible = (query_field.amplitude + doc_field.amplitude) / 2
        interference_ratio = superposed.amplitude / max(max_possible, 1e-6)

        return interference_ratio


# ============================================================================
# CRYSTALLINE RAG SYSTEM
# ============================================================================

class CrystallineRAG:
    """
    Complete RAG system using Crystalline dual-track computation

    Features:
    - Physics-guided retrieval using field operations
    - Maintains semantic lineage throughout pipeline
    - Explainable similarity through dual-track analysis
    - Golden angle spacing for optimal document organization
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.processor = CrystallineTextProcessor(chunk_size, chunk_overlap)
        self.embedder = CrystallineEmbedder()
        self.vector_store = CrystallineVectorStore()
        self.similarity = CrystallineSimilarity()
        self.documents: Dict[str, Document] = {}

    def add_document(self, doc_id: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Add document to RAG system

        Pipeline:
        1. Create document object
        2. Chunk using golden angle spacing
        3. Generate field embeddings for each chunk
        4. Store in vector store
        """
        print(f"Adding document: {doc_id}")

        # Create document
        doc = Document(
            doc_id=doc_id,
            content=content,
            metadata=metadata or {}
        )

        # Chunk document
        chunks = self.processor.chunk_document(doc)
        print(f"  Created {len(chunks)} chunks")

        # Generate embeddings
        embeddings = []
        for idx, chunk in enumerate(chunks):
            embedding = self.embedder.embed_chunk(chunk, idx)
            chunk.field_state = embedding
            embeddings.append(embedding)

        print(f"  Generated {len(embeddings)} embeddings")

        # Store chunks
        self.vector_store.add_chunks(chunks, embeddings)
        doc.chunks = chunks
        self.documents[doc_id] = doc

        print(f"  Document '{doc_id}' added successfully")

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        use_interference: bool = True
    ) -> List[RetrievalResult]:
        """
        Query the RAG system with dual-track retrieval

        Pipeline:
        1. Transform query into field representation
        2. Compute similarity with all chunks (dual-track)
        3. Rank by combined numeric score
        4. Return top-k with full semantic lineage
        """
        print(f"\nQuerying: '{query_text}'")

        # Stage 1: Embed query
        query_field = self.embedder.embed_text(query_text, chunk_idx=-1)
        print(f"  Query field: domain={query_field.domain.name}, phase={query_field.phase:.1f}°, amp={query_field.amplitude:.3f}")

        # Stage 2: Compute similarities
        results = []
        all_chunks = self.vector_store.get_all_chunks()

        for chunk, chunk_field in all_chunks:
            # Standard similarity
            sim_metrics = self.similarity.compute_similarity(query_field, chunk_field)

            # Optional: Use interference scoring
            if use_interference:
                interference_score = self.similarity.compute_interference_score(query_field, chunk_field)
                # Blend standard and interference scores
                final_score = 0.7 * sim_metrics['numeric_score'] + 0.3 * interference_score
            else:
                final_score = sim_metrics['numeric_score']

            result = RetrievalResult(
                chunk=chunk,
                numeric_score=final_score,
                semantic_lineage=chunk_field.meaning,
                coherence=sim_metrics['coherence'],
                phase_alignment=sim_metrics['phase_alignment']
            )
            results.append(result)

        # Stage 3: Rank and return top-k
        results.sort(key=lambda r: r.numeric_score, reverse=True)
        top_results = results[:top_k]

        print(f"  Found {len(results)} chunks, returning top {top_k}")

        return top_results

    def generate_response(
        self,
        query_text: str,
        retrieval_results: List[RetrievalResult],
        max_context_chunks: int = 3
    ) -> Dict[str, Any]:
        """
        Generate response using retrieved context

        Uses field_superpose to combine multiple chunks into coherent context
        """
        # Take top chunks for context
        context_chunks = retrieval_results[:max_context_chunks]

        # Superpose chunk fields to create unified context representation
        if context_chunks:
            chunk_fields = [r.chunk.field_state for r in context_chunks if r.chunk.field_state]
            weights = [r.numeric_score for r in context_chunks]

            # Normalize weights
            weight_sum = sum(weights)
            weights = [w / weight_sum for w in weights]

            # Create context field via superposition
            context_field = field_superpose(chunk_fields, weights=weights, tag='context')

            # Transform to OUTPUT domain
            output_field = field_transform(
                context_field,
                domain=Domain.OUTPUT,
                shell=2,
                phase=180.0,
                curvature=-2.0,
                tag='response'
            )

            # Generate response structure
            response = {
                'query': query_text,
                'answer': {
                    'context_chunks': [
                        {
                            'content': r.chunk.content[:200] + '...',
                            'doc_id': r.chunk.doc_id,
                            'score': round(r.numeric_score, 4),
                            'phase_alignment': round(r.phase_alignment, 4)
                        }
                        for r in context_chunks
                    ],
                    'synthesis': f"Based on {len(context_chunks)} relevant chunks with combined coherence of {output_field.coherence:.3f}"
                },
                'field_analysis': {
                    'output_domain': output_field.domain.name,
                    'semantic_lineage': output_field.meaning,
                    'coherence': round(output_field.coherence, 4),
                    'phase': round(output_field.phase, 2),
                    'amplitude': round(output_field.amplitude, 4)
                },
                'retrieval_quality': {
                    'avg_score': round(sum(r.numeric_score for r in context_chunks) / len(context_chunks), 4),
                    'avg_coherence': round(sum(r.coherence for r in context_chunks) / len(context_chunks), 4),
                    'total_chunks_considered': len(retrieval_results)
                }
            }
        else:
            response = {
                'query': query_text,
                'answer': {'synthesis': 'No relevant context found'},
                'field_analysis': {},
                'retrieval_quality': {'total_chunks_considered': 0}
            }

        return response

    def query_and_generate(
        self,
        query_text: str,
        top_k: int = 5,
        max_context_chunks: int = 3
    ) -> Dict[str, Any]:
        """
        Complete RAG pipeline: query + generate response
        """
        # Retrieve
        results = self.query(query_text, top_k=top_k)

        # Generate
        response = self.generate_response(query_text, results, max_context_chunks)

        return response


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def demo_crystalline_rag():
    """Demonstrate the Crystalline RAG system"""
    print("="*80)
    print("CRYSTALLINE RAG DEMONSTRATION")
    print("Physics-Guided Retrieval-Augmented Generation")
    print("="*80)

    # Initialize RAG
    rag = CrystallineRAG(chunk_size=100, chunk_overlap=20)

    # Add sample documents
    doc1 = """
    Crystalline is a domain-specific language for physics-guided code synthesis.
    It uses geometric field optimization and evolutionary transformations to
    generate optimal code. The system maintains dual-track computation where
    every value carries both numeric state and semantic meaning throughout
    execution. This enables deterministic, explainable code generation.
    """

    doc2 = """
    Retrieval-Augmented Generation (RAG) is a technique that combines
    information retrieval with language generation. It retrieves relevant
    documents from a knowledge base and uses them as context for generating
    responses. RAG systems typically use vector embeddings and similarity
    search to find relevant content.
    """

    doc3 = """
    Field theory in physics describes how fields vary in space and time.
    Electromagnetic fields, gravitational fields, and quantum fields are
    all examples. In Crystalline, we use field theory as a computational
    metaphor, treating program structure as an electromagnetic field with
    amplitude, phase, and curvature properties.
    """

    rag.add_document("doc1_crystalline", doc1, {"topic": "crystalline", "type": "technical"})
    rag.add_document("doc2_rag", doc2, {"topic": "rag", "type": "technical"})
    rag.add_document("doc3_physics", doc3, {"topic": "physics", "type": "educational"})

    print("\n" + "="*80)
    print("QUERY 1: What is Crystalline?")
    print("="*80)

    response1 = rag.query_and_generate(
        "What is Crystalline and how does it work?",
        top_k=5,
        max_context_chunks=2
    )

    print("\n--- Response ---")
    print(f"Query: {response1['query']}")
    print(f"\nSynthesis: {response1['answer']['synthesis']}")
    print(f"\nTop Contexts:")
    for i, ctx in enumerate(response1['answer']['context_chunks'], 1):
        print(f"\n{i}. [Doc: {ctx['doc_id']}] (Score: {ctx['score']}, Phase Align: {ctx['phase_alignment']})")
        print(f"   {ctx['content']}")

    print(f"\nField Analysis:")
    for key, value in response1['field_analysis'].items():
        print(f"  {key}: {value}")

    print(f"\nRetrieval Quality:")
    for key, value in response1['retrieval_quality'].items():
        print(f"  {key}: {value}")

    print("\n" + "="*80)
    print("QUERY 2: How do fields work?")
    print("="*80)

    response2 = rag.query_and_generate(
        "How do fields work in physics and computation?",
        top_k=5,
        max_context_chunks=2
    )

    print("\n--- Response ---")
    print(f"Query: {response2['query']}")
    print(f"\nSynthesis: {response2['answer']['synthesis']}")
    print(f"\nTop Contexts:")
    for i, ctx in enumerate(response2['answer']['context_chunks'], 1):
        print(f"\n{i}. [Doc: {ctx['doc_id']}] (Score: {ctx['score']}, Phase Align: {ctx['phase_alignment']})")
        print(f"   {ctx['content']}")

    print("\n" + "="*80)
    print("KEY FEATURES OF CRYSTALLINE RAG:")
    print("="*80)
    print("✓ Dual-track embeddings: numeric vectors + semantic meaning")
    print("✓ Physics-guided similarity: phase alignment, field coupling")
    print("✓ Golden angle spacing: optimal document organization")
    print("✓ Explainable retrieval: full semantic lineage preserved")
    print("✓ Field superposition: coherent multi-document synthesis")
    print("="*80)


if __name__ == "__main__":
    demo_crystalline_rag()
