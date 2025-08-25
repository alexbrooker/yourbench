# Multilingual Text Chunking Support

## Overview

YouRBench now supports customizable sentence-based text chunking with configurable delimiters, enabling proper text processing for multiple languages including Chinese, Japanese, and other languages that use different punctuation marks.

## Features

- **Flexible Chunking Modes**: Choose between token-based or sentence-based chunking
- **Customizable Delimiters**: Configure sentence delimiters for any language
- **Overlap Support**: Control overlap between chunks for better context preservation
- **Minimum Length Control**: Ensure chunks meet minimum length requirements

## Configuration

### Basic Configuration

In your YAML configuration file, set the chunking mode and delimiters:

```yaml
pipeline_config:
  chunking:
    run: true
    chunking_mode: "sentence"  # or "token" for token-based chunking
    sentence_delimiters: "[.!?]"  # English delimiters (default)
```

### Chinese Language Configuration

For Chinese text, use Chinese punctuation marks:

```yaml
pipeline_config:
  chunking:
    chunking_mode: "sentence"
    sentence_delimiters: "[\u3002\uff01\uff1f]"  # 。！？
    max_sentences_per_chunk: 10
    sentence_overlap: 2
```

### Mixed Language Support

For documents containing both English and Chinese:

```yaml
pipeline_config:
  chunking:
    chunking_mode: "sentence"
    sentence_delimiters: "[.!?\u3002\uff01\uff1f]"  # English and Chinese
```

## Delimiter Reference

### Common Language Delimiters

| Language | Delimiters | Unicode | Pattern |
|----------|------------|---------|---------|
| English | . ! ? | - | `[.!?]` |
| Chinese | 。！？ | \u3002 \uff01 \uff1f | `[\u3002\uff01\uff1f]` |
| Japanese | 。！？ | \u3002 \uff01 \uff1f | `[\u3002\uff01\uff1f]` |
| Arabic | . ! ؟ | - \u061f | `[.!\u061f]` |
| Spanish | . ! ? ¡ ¿ | - \u00a1 \u00bf | `[.!?\u00a1\u00bf]` |

### Configuration Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `chunking_mode` | Choose between "token" or "sentence" | "token" | - |
| `max_sentences_per_chunk` | Maximum sentences per chunk | 10 | 1-100 |
| `sentence_overlap` | Number of overlapping sentences | 2 | 0+ |
| `sentence_delimiters` | Regex pattern for delimiters | `[.!?]` | Any regex |
| `min_chunk_length` | Minimum characters per chunk | 100 | 10+ |

## Examples

### Example 1: Processing Chinese Documentation

```yaml
dataset_config:
  dataset_name: "chinese-docs"
  
pipeline_config:
  chunking:
    run: true
    chunking_mode: "sentence"
    sentence_delimiters: "[\u3002\uff01\uff1f]"
    max_sentences_per_chunk: 8
    sentence_overlap: 1
    min_chunk_length: 50
```

### Example 2: Processing Mixed Content

```yaml
pipeline_config:
  chunking:
    run: true
    chunking_mode: "sentence"
    # Supports English, Chinese, Japanese punctuation
    sentence_delimiters: "[.!?\u3002\uff01\uff1f]"
    max_sentences_per_chunk: 12
    sentence_overlap: 2
```

## Backward Compatibility

The default configuration remains token-based chunking for backward compatibility. Existing configurations will continue to work without modification.

## Performance Considerations

- **Sentence-based chunking** is generally faster for processing but may create chunks of varying token sizes
- **Token-based chunking** provides precise token control but requires tokenization overhead
- Choose based on your specific requirements for chunk size consistency vs processing speed

## Troubleshooting

### Issue: Sentences not splitting correctly
- Verify your delimiter pattern matches the punctuation in your text
- Check for Unicode encoding issues
- Test your pattern with a small sample first

### Issue: Chunks too short/long
- Adjust `max_sentences_per_chunk` parameter
- Consider using `min_chunk_length` to merge short chunks
- For token control, switch to token-based chunking mode