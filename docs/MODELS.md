# Models and sampling settings

The benchmark compares small (2–4B) open models as each vendor intends them to
be run, not under one uniform decoding setup. Temperature, nucleus settings and
reasoning mode therefore **differ across models by design**. Any between-model
comparison inherits these differences; treat them as part of what "the model"
means here, and check results against the caveats below before attributing a
difference to the model itself.

The source of truth is `sweeps/models.yaml`. Each Modal run also writes
`<tag>/<model>/model.json` with the exact settings, vLLM version and CUDA image
used, and every game's `config.yaml` records the request parameters it was
played with.

## Settings (vLLM 0.31.0)

| Model | Reasoning | Temperature | Other sampling | Max output tokens | Source of sampling values |
|---|---|---|---|---|---|
| MiniCPM5-2B | yes (`<think>`) | 1.0 | top_p 0.95 | 16,384 | model card + `generation_config.json` |
| G9v3-3B | yes (`<think>`) | 0.9 | top_p 0.95 | 16,384 | `generation_config.json` (card gives none) |
| K2-Horizon-3.7B | yes, `reasoning_effort: medium` | 1.0 | top_p 0.95 | 16,384 | model card; **effort lowered from the recommended high** (our choice, for cost) |
| Granite-4.2-3B | yes (`<think>`) | 1.0 | top_p 0.95 | 16,384 | model card ("required for all modes") |
| Nanbeige4.1-3B | yes (`<think>`) | 0.6 | top_p 0.95 | 16,384 | model card |
| LFM2.5-2.6B | yes (always thinks) | 0.1 | top_k 50, repetition_penalty 1.1 | 16,384 | model card + `generation_config.json` |
| Phi-4-mini-instruct | **no** (no thinking mode) | 0.7 | — | 2,048 | **our choice**: the card shows 0.0 only in code examples |
| Qwen3.5-4B (baseline, not in the main run) | yes (`<think>`) | 1.0 | top_p 0.95, top_k 20, presence_penalty 1.5 | 16,384 | model card, thinking mode for general tasks |

## Caveats for analysis

1. **Reasoning vs non-reasoning.** Phi-4-mini is the only non-reasoning model.
   In the check run it produced about 60 tokens per response against roughly
   2,000–7,000 for the others. Differences between Phi-4-mini and the rest
   confound model quality with deliberation. Report it separately or as a
   non-reasoning reference, not as one point on the same scale.
2. **Temperature spread (0.1 to 1.0).** Higher temperature adds behavioural
   variance across seeds and can by itself make tacit coordination (repeating
   a focal price) less stable. LFM2.5 at 0.1 is close to greedy, so its seeds
   vary mainly through the environment draws. Compare seed-to-seed variance
   with this in mind before reading it as model "stability".
3. **Unequal reasoning budgets.** All reasoning models share the 16k output cap,
   but they use it differently (check run means: LFM2.5 ~1,900 tokens/response,
   K2 ~4,200, Nanbeige ~5,100, G9v3 ~5,100, MiniCPM5 ~5,900, Granite ~7,000 at
   an 8k cap). K2 runs below its recommended effort. Truncation (`finish_reason
   == "length"`) ends a turn without an action, so check each model's
   `performance.truncated_rate` and `missing_action_rate` before comparing
   collusion indices.
4. **Tool-calling path.** Phi-4-mini often writes calls as text that vLLM's
   parser misses. agora recovers them and reports
   `performance.text_recovered_rate`. A model that needs recovery has an
   easier-to-fail interface than one with native calls.
5. **Recovery-parser version differs between runs.** In `main-20261025` the six
   reasoning models ran with JSON-only text recovery (they made native tool
   calls in the check run, so recovery should rarely matter for them). Phi-4-mini's
   games there missed 91% of GridStorage and 54% of RideHailing actions because it
   writes calls as code (`hold`, `drive_in_zone(zone: 2)`, `charge 5`), so it was
   rerun as `main-20261025-phi-r2` with code-style recovery (commit "Recover
   code-style tool calls"). Use the r2 tag for Phi-4-mini. Before final analysis,
   replay each reasoning model's no-call replies through the current parser
   and rerun any model where it would recover a non-trivial share.
6. **Related models.** G9v3-3B shares MiniCPM5's tokenizer, chat template and
   layer widths (52 vs 42 layers) and is probably a MiniCPM5 derivative. Do
   not count the two as independent observations of "small models".

## Changing settings

To test sensitivity, add a model entry with the same `hf_id` and a different
`name` (e.g. `granite-4.2-3b-t0.7`) and override `params`. Results land under
the new name, so variants never mix with the main run.
