# CLI

```bash
python -m lemmas <subcommand> "<query>" [flags]
```

Works offline against the deterministic echo stub OR with `OPENAI_API_KEY`
against `gpt-4o-mini`. All subcommands accept `--json` for piping.

## Subcommands

```bash
python -m lemmas cove "Who invented the laser?" --n-questions 4
python -m lemmas self_consistency "What is 13 * 17?" --n 5 --extractor last_number
python -m lemmas best_of_n "Write a haiku" --n 5 --keywords "snow,arctic"
python -m lemmas drift "hello world"          # requires OPENAI_API_KEY
```

## Common flags

| Flag | Default | Notes |
|---|---|---|
| `--model` | `gpt-4o-mini` | Only used with `OPENAI_API_KEY`. |
| `--temperature` | `0.7` | |
| `--max-tokens` | `512` | |
| `--json` | off | Emit JSON instead of human-readable output. |

## Per-subcommand flags

| Subcommand | Flag | Notes |
|---|---|---|
| `cove` | `--n-questions N` | Verification questions to plan (default 4). |
| `self_consistency` | `--n N` | Sample count. |
| `self_consistency` | `--extractor X` | `last_line`, `last_number`, `regex`, `similarity`. |
| `self_consistency` | `--regex PATTERN` | Required when `--extractor regex`. |
| `best_of_n` | `--n N` | Sample count. |
| `best_of_n` | `--target-chars N` | For default `length_scorer`. |
| `best_of_n` | `--keywords k1,k2` | Use `keyword_scorer` instead. |
| `drift` | `--bucket NAME` | Bucket id (default `default`). |
| `drift` | `--z-threshold F` | Drift-flag threshold (default 3.0). |
| `drift` | `--warmup-n N` | Observations before drift can fire (default 20). |
