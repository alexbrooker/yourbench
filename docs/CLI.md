# YourBench CLI Reference

YourBench provides a rich command-line interface for generating evaluation datasets from your documents.

## Installation

```bash
# Install with uv (recommended)
uv pip install yourbench

# Or run directly without installing
uvx --from yourbench yourbench --help
```

## Commands Overview

| Command | Description |
|---------|-------------|
| `run` | Run the full pipeline with a config file |
| `validate` | Check a config file without running |
| `estimate` | Estimate token usage before running |
| `init` | Generate a starter config interactively |
| `stages` | List all available pipeline stages |
| `version` | Show YourBench version |

---

## `yourbench run`

Run the YourBench pipeline with a configuration file.

```bash
yourbench run <config_path> [OPTIONS]
```

**Arguments:**
- `config_path` - Path to your YAML configuration file (required)

**Options:**
- `--debug, -d` - Enable debug logging (shows detailed progress)
- `--quiet, -q` - Minimal output (only errors)
- `--no-banner` - Hide the startup banner

**Examples:**

```bash
# Basic run
yourbench run config.yaml

# With debug output
yourbench run config.yaml --debug

# Quiet mode for scripts
yourbench run config.yaml --quiet
```

**Output:**
- Progress bars for each pipeline stage
- Token usage statistics per stage
- Final dataset location (Hub URL or local path)

---

## `yourbench validate`

Validate a configuration file without running the pipeline. Useful for catching errors before a long run.

```bash
yourbench validate <config_path>
```

**Arguments:**
- `config_path` - Path to YAML config file to validate (required)

**Examples:**

```bash
yourbench validate config.yaml
```

**Output:**

```
✓ Configuration is valid!

                             Configuration Summary                              
┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Setting     ┃ Value                                                          ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Dataset     │ my-benchmark                                                   │
│ Push to Hub │ ✓                                                              │
│ Private     │ ✗                                                              │
│ Models      │ openai/gpt-4o-mini                                             │
│ Stages      │ ingestion, summarization, chunking, ...                        │
└─────────────┴────────────────────────────────────────────────────────────────┘

Enabled stages (5):
  1. ingestion
  2. summarization
  3. chunking
  4. single_hop_question_generation
  5. prepare_lighteval
```

**Checks performed:**
- YAML syntax validity
- Required fields present
- Model configuration correct
- Stage dependencies satisfied
- Environment variables resolved

---

## `yourbench estimate`

Estimate token usage for a pipeline run before executing it. Helps with cost planning.

```bash
yourbench estimate <config_path>
```

**Arguments:**
- `config_path` - Path to YAML config file (required)

**Examples:**

```bash
yourbench estimate config.yaml
```

**Output:**

```
Source Documents:
  Files: 3
  Estimated tokens: 15.2K

                           Token Estimation by Stage                            
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Stage           ┃ Input Tokens ┃ Output Tokens ┃ API Calls ┃ Notes           ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ Ingestion       │            - │             - │         - │ No LLM calls    │
│ Summarization   │         4.5K │          6.0K │         3 │                 │
│ Chunking        │            - │             - │         - │ No LLM calls    │
│ Single Hop QG   │        27.6K │          4.5K │         3 │                 │
└─────────────────┴──────────────┴───────────────┴───────────┴─────────────────┘

╭─────── Summary ────────╮
│ Total Estimated Usage: │
│   Input tokens:  32.1K │
│   Output tokens: 10.5K │
│   Total:         42.6K │
╰────────────────────────╯
```

**Notes:**
- Estimates use tiktoken for accurate token counting
- Actual usage may vary based on model responses
- Stages without LLM calls (ingestion, chunking) show "-"

---

## `yourbench init`

Generate a starter configuration file interactively.

```bash
yourbench init [OPTIONS]
```

**Options:**
- `--output, -o` - Output file path (default: `config.yaml`)
- `--force, -f` - Overwrite existing file without prompting

**Examples:**

```bash
# Create config.yaml in current directory
yourbench init

# Create with custom name
yourbench init -o my-project/config.yaml

# Overwrite existing
yourbench init -o config.yaml --force
```

**Interactive prompts:**
1. Dataset name for HuggingFace Hub
2. Model provider (OpenAI, HuggingFace, local vLLM, custom)
3. Source documents directory
4. Pipeline stages to enable
5. Output preferences (Hub push, local save)

---

## `yourbench stages`

Display all available pipeline stages with descriptions.

```bash
yourbench stages
```

**Output:**

```
                                Pipeline Stages                                 
┏━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ #   ┃ Stage                              ┃ Description                       ┃
┡━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 1   │ ingestion                          │ Process source documents          │
│ 2   │ summarization                      │ Generate summaries                │
│ 3   │ chunking                           │ Split into chunks                 │
│ 4   │ single_hop_question_generation     │ Generate standalone Q&A pairs     │
│ 5   │ multi_hop_question_generation      │ Multi-chunk questions             │
│ 6   │ cross_document_question_generation │ Cross-document questions          │
│ 7   │ question_rewriting                 │ Rewrite for clarity               │
│ 8   │ prepare_lighteval                  │ Format for LightEval              │
│ 9   │ citation_score_filtering           │ Filter by citation quality        │
└─────┴────────────────────────────────────┴───────────────────────────────────┘
```

**Stage details:**

| Stage | LLM Required | Description |
|-------|--------------|-------------|
| `ingestion` | No | Parse PDFs, Word docs, HTML into Markdown |
| `summarization` | Yes | Generate document summaries |
| `chunking` | No | Split documents into semantic chunks |
| `single_hop_question_generation` | Yes | Q&A pairs from individual chunks |
| `multi_hop_question_generation` | Yes | Questions requiring multiple chunks |
| `cross_document_question_generation` | Yes | Questions spanning documents |
| `question_rewriting` | Yes | Improve question clarity |
| `prepare_lighteval` | No | Format for evaluation framework |
| `citation_score_filtering` | No | Filter low-quality citations |

---

## `yourbench version`

Show the installed YourBench version.

```bash
yourbench version
```

**Output:**

```
YourBench v0.9.0
```

---

## Environment Variables

The CLI respects these environment variables (can also be set in `.env`):

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | HuggingFace token for Hub operations |
| `HF_ORGANIZATION` | Default organization for dataset uploads |
| `OPENAI_API_KEY` | OpenAI API key |
| `OPENAI_BASE_URL` | Custom OpenAI-compatible endpoint |
| `OPENAI_MODEL` | Default model name |

Use `$VAR_NAME` syntax in config files to reference environment variables:

```yaml
model_list:
  - model_name: $OPENAI_MODEL
    api_key: $OPENAI_API_KEY
    base_url: $OPENAI_BASE_URL
```

---

## Workflow Example

Typical workflow for generating a benchmark:

```bash
# 1. Generate starter config
yourbench init -o my-benchmark/config.yaml

# 2. Edit config as needed
vim my-benchmark/config.yaml

# 3. Validate before running
yourbench validate my-benchmark/config.yaml

# 4. Estimate costs
yourbench estimate my-benchmark/config.yaml

# 5. Run the pipeline
yourbench run my-benchmark/config.yaml --debug
```

---

## Troubleshooting

**"Config validation failed"**
- Run `yourbench validate config.yaml` for detailed error messages
- Check that all required environment variables are set

**"No documents found"**
- Verify `source_documents_dir` path exists
- Check file extensions are supported (.pdf, .md, .txt, .docx, .html)

**"API rate limit exceeded"**
- Reduce `max_concurrent_requests` in model config
- Add delays between runs

**"Token limit exceeded"**
- Use `yourbench estimate` to check token usage
- Reduce chunk size or number of questions per chunk

See [FAQ](./FAQ.md) for more troubleshooting tips.
