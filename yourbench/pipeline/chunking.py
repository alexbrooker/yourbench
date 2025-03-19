"""
chunking.py

Implements the Semantic Chunking stage of the YourBench pipeline using
the `intfloat/multilingual-e5-large-instruct` model for embedding-based
similarity computation. This module also generates plots of sentence-pair
similarities and saves them in the `plots/` folder for debugging and
analysis purposes.

References (from your paper):
- Section 2.2.3 on Semantic Chunking
- Equation for similarity-based boundary detection
- Multi-hop chunking approach
- Now uses E5 embeddings for semantic similarity
"""

import os
import random
import torch
import multiprocessing
import matplotlib.pyplot as plt

from torch.amp import autocast

from typing import Dict, Any, List
from loguru import logger

from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

from yourbench.utils.dataset_engine import custom_load_dataset
from yourbench.utils.dataset_engine import custom_save_dataset

# Parallel Processing 
torch.set_num_threads(multiprocessing.cpu_count())

# GLOBALS 
E5_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
os.makedirs("plots", exist_ok=True)  # Ensure plots folder exists for saving graphs

def run(config: Dict[str, Any]) -> None:
    """
    Run the chunking stage of the pipeline
    """
    chunking_cfg = config.get("pipeline", {}).get("chunking", {})
    if not chunking_cfg.get("run", False):
        logger.info("Chunking stage is disabled. Skipping")
        return

    logger.info("Running chunking stage with E5 embeddings...")

    # Step 1: Load dataset 
    dataset = custom_load_dataset(config=config, step_name="summarization")

    # Step 2: Retrieve chunking parameters 
    cparams = chunking_cfg.get("chunking_configuration", {})
    l_min_tokens = cparams.get("l_min_tokens", 256)
    l_max_tokens = cparams.get("l_max_tokens", 1024)
    tau_threshold = cparams.get("tau_threshold", 0.3)
    # For multi-hop chunking:
    h_min = cparams.get("h_min", 2)
    h_max = cparams.get("h_max", 3)

    logger.debug(
        "Chunking configuration: l_min_tokens={}, l_max_tokens={}, tau_threshold={}, h_min={}, h_max={}",
        l_min_tokens, l_max_tokens, tau_threshold, h_min, h_max
    )

    # Load E5 model/tokenizer once (GPU-accelerated if available) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(E5_MODEL_NAME)
    model = AutoModel.from_pretrained(E5_MODEL_NAME).to(device).eval()

    # Step 3: Perform chunking on each row (document) 
    all_single_hop_chunks = []
    all_multihop_chunks = []

    for idx, row in enumerate(dataset):
        doc_text = row["document_text"]
        if not doc_text or not doc_text.strip():
            logger.warning("Document at index {} has empty text. Storing empty chunks", idx)
            all_single_hop_chunks.append([])
            all_multihop_chunks.append([])
            continue

        sentences = _split_into_sentences(doc_text)

        # If the document is extremely short or fails to produce sentences
        if not sentences:
            logger.warning("No valid sentences found for doc at index {}", idx)
            all_single_hop_chunks.append([])
            all_multihop_chunks.append([])
            continue

        # 3A: Compute embeddings for each sentence 
        # E5 does not require instructions for the document side
        # so we pass the raw sentences
        sentence_embeddings = _compute_embeddings(tokenizer, model, sentences, device)

        # Vectorized computation of cosine similarity
        if len(sentence_embeddings) > 1:
            sentence_embeddings = torch.stack(sentence_embeddings)  # a list of tensors to a single tensor
            similarities = F.cosine_similarity(sentence_embeddings[:-1], sentence_embeddings[1:], dim=1).tolist()
        else:
            similarities = []  # If only one sentence, no similarities to compute

        # 3C: Generate a plot of these similarities for inspection 
        _plot_sentence_similarities(similarities, idx)

        # 3D: Single-hop chunking with boundary logic 
        single_hop = _chunk_document(
            sentences,
            similarities,
            l_min_tokens,
            l_max_tokens,
            tau_threshold
        )

        # 3E: Multi-hop chunking 
        multihop = _multihop_chunking(single_hop, h_min, h_max)

        all_single_hop_chunks.append(single_hop)
        all_multihop_chunks.append(multihop)

    # Step 4: Add new columns and save 
    dataset = dataset.add_column("chunks", all_single_hop_chunks)
    dataset = dataset.add_column("multihop_chunks", all_multihop_chunks)
    dataset = dataset.add_column("chunking_model", [E5_MODEL_NAME] * len(dataset))

    # Step 5: Save dataset 
    logger.info("Saving chunked subset to HF")
    custom_save_dataset(
        dataset=dataset,
        config=config,
        step_name="chunking",
    )
    logger.success("Chunking stage complete")


def _compute_embeddings(
    tokenizer: AutoTokenizer,
    model: AutoModel,
    texts: List[str],
    device: torch.device,
    batch_size: int = 16,
    max_len: int = 512
) -> List[torch.Tensor]:
    """
    Compute sentence embeddings in batches using all available computational resources
    Uses AMP on CUDA for faster execution
    """
    embeddings = []
    model.eval()

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_dict = tokenizer(
            batch_texts,
            max_length=max_len,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            with autocast("cuda" if torch.cuda.is_available() else "cpu"):
                outputs = model(**batch_dict)
                last_hidden = outputs.last_hidden_state
                attention_mask = batch_dict["attention_mask"]
                last_hidden = last_hidden.masked_fill(~attention_mask[..., None].bool(), 0.0)
                batch_embeddings = last_hidden.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)

        embeddings.extend(batch_embeddings.cpu())

    return embeddings

def _split_into_sentences(text: str) -> List[str]:
    """
    Naive sentence splitting by '.', '!', or '?'.
    This is a placeholder. For more robust usage, consider spaCy or NLTK-based splitting.
    """
    import re
    # Replace newlines with spaces
    text = text.replace("\n", " ").strip()
    if not text:
        return []

    # Split on . ! ?
    segments = re.split(r'([.!?])', text)
    sentences = []
    for i in range(0, len(segments), 2):
        if i + 1 < len(segments):
            s = (segments[i] + segments[i + 1]).strip()
        else:
            s = segments[i].strip()
        if s:
            sentences.append(s)
    return sentences


def _chunk_document(
    sentences: List[str],
    similarities: List[float],
    l_min_tokens: int,
    l_max_tokens: int,
    tau: float
) -> List[str]:
    """
    Split a document into semantic chunks based on E5-based sentence similarities
    and token-length constraints
    """
    chunks = []
    current_chunk = []
    current_len = 0  # running tally of tokens in current chunk

    n_sents = len(sentences)
    for i in range(n_sents):
        s = sentences[i]
        s_tokens = s.split()
        s_len = len(s_tokens)

        # If adding this sentence alone exceeds l_max, isolate it as a single chunk
        if s_len >= l_max_tokens:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            chunks.append(s)
            current_chunk = []
            current_len = 0
            continue

        # Add the sentence to the current chunk
        current_chunk.append(s)
        current_len += s_len

        # Check if we need to start a new chunk
        # Condition A: Exceeds l_max
        if current_len >= l_max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_len = 0
            continue

        # Condition B: If we have at least l_min, check the similarity to the next sentence
        if current_len >= l_min_tokens and i < n_sents - 1:
            # similarity for pair (i, i+1) is similarities[i]
            if similarities[i] < tau:
                # boundary
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_len = 0

    # If leftover sentences remain in current_chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def _multihop_chunking(
    single_hop_chunks: List[str],
    h_min: int,
    h_max: int
) -> List[str]:
    """
    Create multi-hop chunks by randomly sampling subsets of single-hop chunks
    and concatenating them. (Per Section 2.2.3 in the paper.)
    """
    if not single_hop_chunks:
        return []

    k = random.randint(h_min, h_max)
    k = min(k, len(single_hop_chunks))
    sampled_indices = sorted(random.sample(range(len(single_hop_chunks)), k))
    multi_hop_concat = " ".join(single_hop_chunks[i] for i in sampled_indices)

    return [multi_hop_concat]


def _plot_sentence_similarities(similarities: List[float], doc_idx: int) -> None:
    """
    Plot and save the distribution of consecutive sentence similarities for a given document.
    """
    if not similarities:
        return

    plt.figure(figsize=(8, 4))
    plt.plot(range(len(similarities)), similarities, marker='o')
    plt.title(f"Consecutive Sentence Similarities (doc {doc_idx})")
    plt.xlabel("Sentence Pair Index (i -> i+1)")
    plt.ylabel("Cosine Similarity (E5 Embeddings)")
    plt.ylim([0.0, 1.05])
    plt.grid(True)
    plot_path = os.path.join("plots", f"chunking_document_{doc_idx}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.debug("Saved similarity plot for document {} at '{}'", doc_idx, plot_path)
