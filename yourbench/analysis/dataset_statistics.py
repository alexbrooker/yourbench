# yourbench/analysis/dataset_analysis.py
# =============================================================================
# dataset_analysis.py
# =============================================================================
"""
Analyze the distribution of chunk lengths and document lengths in the 'chunked_documents' subset,
generating publication-quality log-CDF plots.

Refactored from user-provided code to integrate with the YourBench pipeline. 
We read the 'chunked_documents' subset which contains both chunks and document text,
compute their character lengths, and plot separate log-CDFs,
saving the resulting figures in both PDF and PNG formats.

Usage (CLI):
    yourbench analyze dataset_analysis
"""

import os
import copy
import math
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from typing import List, Tuple, Dict, Any
from scipy import stats

# YourBench utilities
from yourbench.utils.dataset_engine import smart_load_dataset
from yourbench.utils.loading_engine import load_config
from yourbench.config_cache import get_last_config


def _calculate_cdf(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sorts the data and computes empirical CDF.

    Args:
        data (np.ndarray): 1D array of values (lengths, etc.)

    Returns:
        data_sorted (np.ndarray): Sorted data.
        p (np.ndarray): Corresponding cumulative probabilities.
    """
    data_sorted = np.sort(data)
    p = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
    return data_sorted, p


def _plot_distribution(lengths: np.ndarray, title: str, filename_prefix: str, plot_dir: str, use_log_scale: bool = False) -> None:
    """
    Create and save a publication-quality CDF plot for length distribution.
    
    Args:
        lengths (np.ndarray): Array of length values to plot
        title (str): Title for the plot
        filename_prefix (str): Prefix for the output filenames
        plot_dir (str): Directory to save plots
        use_log_scale (bool): Whether to use logarithmic scale for x-axis
        
    Returns:
        None
    """
    # Some stats for annotation
    median_length = np.median(lengths)
    mean_length = np.mean(lengths)
    sum_length = np.sum(lengths)

    # Safe mode calculation
    try:
        mode_result = stats.mode(lengths, keepdims=True)
        if hasattr(mode_result, 'mode'):
            mode_value = mode_result.mode[0]
        else:
            # older scipy
            mode_value = mode_result[0][0]
    except Exception:
        mode_value = None

    # Fit a log-normal distribution
    shape, loc, scale = stats.lognorm.fit(lengths, floc=0)
    
    # Compute empirical CDF
    sorted_lengths, cdf_values = _calculate_cdf(lengths)
    
    # We'll pick a midpoint for reference annotation
    midpoint_idx = len(sorted_lengths) // 2
    x_reference = sorted_lengths[midpoint_idx]
    y_reference = cdf_values[midpoint_idx]

    # Create figure
    plt.figure(figsize=(4, 4))

    line_color = '#FFD21E'  # from user code
    plt.plot(sorted_lengths, cdf_values, color=line_color, lw=1.8, label='Empirical CDF')
    
    # Apply log scale if requested
    if use_log_scale:
        plt.xscale('log')
        scale_label = "Log scale"
    else:
        scale_label = "Linear scale"
        
    plt.xlabel('n characters')
    plt.ylabel('Cumulative Probability')
    plt.title(f"{title} ({scale_label})", pad=10)

    def format_with_k(num: float) -> str:
        """
        Utility function to format numeric values with 'k' 
        (e.g., 1000 -> '1k', 2800 -> '2.8k')
        """
        if num >= 1000:
            return f'{num/1000:.1f}k'.replace('.0k', 'k')
        else:
            return f'{num:.1f}'.replace('.0', '')

    # Annotate with stats
    annotation_text = (
        f'n = {len(lengths):,}\n'
        f"Mean = {format_with_k(mean_length)}\n"
        f"Median = {format_with_k(median_length)}\n"
        f"sum(n) = {format_with_k(sum_length)}\n"
        f"Log-normal fit: σ={shape:.2f}, μ={math.log(scale):.2f}"
    )

    plt.annotate(
        annotation_text,
        xy=(0.95, 0.05),
        xycoords='axes fraction',
        fontsize=9,
        color='#333333',
        va='bottom',
        ha='right',
        bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='#CCCCCC', alpha=0.8)
    )

    # Create filename suffix based on scale
    scale_suffix = "_log" if use_log_scale else ""
    
    # Save the plot in multiple formats (PDF, PNG)
    vector_path = os.path.join(plot_dir, f"{filename_prefix}{scale_suffix}.pdf")
    plt.savefig(vector_path, format='pdf', dpi=300, bbox_inches='tight')

    raster_path = os.path.join(plot_dir, f"{filename_prefix}{scale_suffix}.png")
    plt.savefig(raster_path, format='png', dpi=500, bbox_inches='tight', transparent=True)
    plt.savefig(raster_path + "_no_transparent.png", format='png', dpi=500, bbox_inches='tight', transparent=False)

    # Add a small text annotation below
    plt.text(
        0, -0.05, 'YourBench Distribution Analysis', 
        transform=plt.gca().transAxes, 
        fontsize=6, 
        color='gray', 
        alpha=0.7
    )

    logger.info(f"Plot saved to:\n- {vector_path} (vector)\n- {raster_path} (raster)")
    plt.close()


def run(*args: str) -> None:
    """
    Main entry point to run the dataset analysis.

    Steps:
      1. Load config (if any) or default to last used config.
      2. Load the 'chunked_documents' subset which contains both chunks and document text.
      3. Create two separate plots:
         - Distribution of chunk lengths
         - Distribution of document lengths
      4. Save plots in PDF and PNG formats.

    Returns:
      None
    """
    logger.info("Running dataset_analysis to measure both chunk and document length distributions...")

    # 1. Attempt to read config
    try:
        config_path = get_last_config()
        if not config_path:
            logger.warning("No cached config found; proceeding with minimal/no config.")
            config = {}
        else:
            config = load_config(config_path)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        config = {}

    # Setup Matplotlib style settings (publication-quality)
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.6,
        'lines.linewidth': 1.5,
        'lines.markersize': 4,
        'savefig.dpi': 300,
        'savefig.format': 'pdf',
        'figure.constrained_layout.use': True,
    })

    # Create the plots directory if it doesn't exist
    plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../plots")
    os.makedirs(plot_dir, exist_ok=True)

    # 2. Load the chunked_documents dataset (contains both chunks and document text)
    dataset_name = config.get("hf_configuration", {}).get("global_dataset_name", "yourbench_dataset")
    subset_name = "chunked_documents"
    
    try:
        ds = smart_load_dataset(dataset_name, config, dataset_subset=subset_name)
        logger.info(f"Loaded '{dataset_name}' subset='{subset_name}' with {len(ds)} rows.")
    except Exception as e:
        logger.error(f"Could not load dataset: {e}")
        return

    if not len(ds):
        logger.warning("Dataset is empty. Nothing to analyze.")
        return

    # 3. Process and plot chunk length distribution
    # Extract chunk texts and compute lengths
    all_chunk_texts: List[str] = []
    for row in ds:
        chunks = row.get("chunks", [])
        if not isinstance(chunks, list):
            continue
        for ch in chunks:
            if isinstance(ch, dict):
                text = ch.get("chunk_text", "")
                if text and isinstance(text, str):
                    all_chunk_texts.append(text)
            elif isinstance(ch, str):
                all_chunk_texts.append(ch)

    chunk_lengths = np.array([len(t) for t in all_chunk_texts], dtype=np.int64)
    chunk_lengths = chunk_lengths[chunk_lengths > 0]  # Filter out zero-length
    
    if len(chunk_lengths) > 1:
        logger.info(f"Plotting distribution of {len(chunk_lengths)} chunks")
        # Linear scale plot
        _plot_distribution(
            chunk_lengths,
            "Chunk Length Distribution",
            "log_cdf_chunk_lengths",
            plot_dir
        )
        
        # Log scale plot
        _plot_distribution(
            chunk_lengths,
            "Chunk Length Distribution",
            "log_cdf_chunk_lengths",
            plot_dir,
            use_log_scale=True
        )
    else:
        logger.warning("Not enough chunk data to create a meaningful plot")

    # 4. Process and plot document length distribution from the same dataset
    # Extract document texts and compute lengths
    document_lengths: List[int] = []
    for row in ds:
        doc_text = row.get("document_text", "")
        if isinstance(doc_text, str) and doc_text:
            document_lengths.append(len(doc_text))
    
    doc_lengths_arr = np.array(document_lengths, dtype=np.int64)
    doc_lengths_arr = doc_lengths_arr[doc_lengths_arr > 0]  # Filter out zero-length
    
    if len(doc_lengths_arr) > 1:
        logger.info(f"Plotting distribution of {len(doc_lengths_arr)} documents")
        # Linear scale plot
        _plot_distribution(
            doc_lengths_arr,
            "Document Length Distribution",
            "log_cdf_document_lengths",
            plot_dir
        )
        
        # Log scale plot
        _plot_distribution(
            doc_lengths_arr,
            "Tempora-0325B Doc Length",
            "log_cdf_document_lengths",
            plot_dir,
            use_log_scale=True
        )
    else:
        logger.warning("Not enough document data to create a meaningful plot")

    logger.success("Dataset analysis completed successfully.")
