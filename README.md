# RTLC^2

RTLC^2 is the open-source multi-module RTL code completion system released with our paper. The public pipeline combines hybrid retrieval, completion-oriented prompt construction, and selective iteration guided by generation signals.

## What Is Included

- `src/`: core completion system modules
- `RTL-Repo/`: adapted RTL-Repo corpus used for external retrieval
- `archive/json/retrieval_cache/`: bundled external retrieval cache for faster startup
- `prompts/`: public prompt assets
- `config/`: configuration placeholders for public use

## Environment

RTLC^2 is intended to run on Linux with Python 3.10.

Before running the system, prepare the following:

- Python 3.10 environment
- dependencies from `requirements.txt`
- the bundled external retrieval corpus in `RTL-Repo/`
- the bundled retrieval cache in `archive/json/retrieval_cache/`
- access to a completion backend, configured through environment variables when needed

If you use an API backend, set:

- `RTLC2_API_BASE`
- `RTLC2_API_KEY`

If you use a local backend, set:

- `RTLC2_LOCAL_MODEL_PATH`

## Installation

Install Python dependencies from `requirements.txt`.

Typical packages include:

- `transformers`
- `datasets`
- `rank_bm25`
- `fuzzywuzzy`
- `python-Levenshtein`

## Minimal Usage

RTLC^2 is usually invoked through the Realbench-Comp black-box adapter. A minimal end-to-end example is:

```bash
python platform/run_completion_adapter.py \
  --adapter adapters/rtlc2_adapter.py:generate_completion \
  --target-file Realbench-Comp_weighted_001/aes/aes_cipher_top/aes_cipher_top__mask0001.v \
  --repo-root . \
  --context-policy recursive_upstream \
  --include-docs auto \
  --request-out /tmp/rtlc2_request.json \
  --response-out /tmp/rtlc2_response.json \
  --print-completion
```

This command:

1. builds a normalized completion request from a masked RTL target
2. runs the RTLC^2 adapter
3. saves the normalized request and completion response for inspection

To score the result on the same sample:

```bash
python evaluation/evaluate_single_completion.py \
  --target-file Realbench-Comp_weighted_001/aes/aes_cipher_top/aes_cipher_top__mask0001.v \
  --response /tmp/rtlc2_response.json \
  --repo-root . \
  --print-result
```

## Workflow

1. Receive a completion request with prefix, suffix, and cross-file context.
2. Run local retrieval over the provided project files.
3. Optionally augment retrieval with external code when local evidence is insufficient.
4. Build a completion-oriented prompt from retrieved code, documents, suffix, guidance, and prefix.
5. Call the configured backend.
6. Optionally trigger a second pass through selective iteration.
7. Return completion-only code.

## Integration

RTLC^2 is intended to be called through the adapter in Realbench-Comp.

The paired benchmark repository is expected to provide:

- masked target files
- repository source trees
- black-box request construction
- single-sample and batch evaluation

Set runtime configuration through environment variables when needed:

- `RTLC2_API_BASE`
- `RTLC2_API_KEY`
- `RTLC2_LOCAL_MODEL_PATH`

The external retrieval cache is shipped so the first run can skip rebuilding the external retrieval index.

## Interface

Input:
- a request object containing target prefix, target suffix, and cross-file context files

Output:
- completion-only Verilog text
