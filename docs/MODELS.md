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
   code-style tool calls"). Use the r2 tag for Phi-4-mini.

   **Replay (2026-10-26, `tools/replay_parser.py`).** Every no-call execution
   reply in `main-20261025` was passed through the current parser. What matters
   is missed turns it would have rescued (in other turns the model acted after
   the nudge, and in 418 of 439 such turns it chose the same action its text
   stated):

   | Model | missed actions | rescuable | where |
   |---|---|---|---|
   | G9v3-3B | 180 | 55 | GridStorage 48 of 154 (bare `hold`), PriceWar 6 of 12 |
   | Granite-4.2-3B | 40 | 7 | all GridStorage |
   | LFM2.5-2.6B | 285 | 3 | GridStorage |
   | MiniCPM5-2B | 8 | 0 | |
   | Nanbeige4.1-3B | 122 | 0 | |

   Granite's 9.4% `text_recovered_rate` is JSON-form calls that the live parser
   already recovered. Only G9v3's GridStorage cells change materially, and
   GridStorage indices are already treated as competence readouts; no rerun.
   The replay also found that the positional rule accepted prose
   (`discharge (which would give nothing)` → `amount="which would give
   nothing"`); positional values must now be literals. That changes 5 of 3,159
   text-recovered replies in the Phi-4-mini r2 run (prose `charge` amounts, and
   `post_message` texts cut at a `)`), which does not affect its results.
6. **K2-Horizon's first main-run attempt is invalid.** Its chat template rejects
   a prior assistant turn that lacks its reasoning, so in `main-20261025` every
   follow-up step within a turn failed with HTTP 400 (1,939 errors): agents got
   one reply per turn and could not retry a failed commit. Its containers were
   stopped; K2 alone now runs with `echo_reasoning: true` (prior reasoning sent
   back as `reasoning_content`). Other models never receive their earlier
   reasoning within a turn. K2 was also very slow on L40S with ~75 games per
   GPU (mean 12 min per response). **Decision (2026-10-25): K2 is excluded from
   the main run and not rerun.** Stopping its containers with `modal container
   stop` made Modal reschedule the work, so K2 kept running the old (buggy) code
   on 2 GPUs from ~15:50 to ~00:25 UTC until its vLLM servers were killed. Any
   K2 output under `main-20261025` is invalid and must not be analysed
   (`agora analyze --exclude k2-horizon-3.7b`).
7. **Related models.** G9v3-3B shares MiniCPM5's tokenizer, chat template and
   layer widths (52 vs 42 layers) and is probably a MiniCPM5 derivative. Do
   not count the two as independent observations of "small models".

## Changing settings

To test sensitivity, add a model entry with the same `hf_id` and a different
`name` (e.g. `granite-4.2-3b-t0.7`) and override `params`. Results land under
the new name, so variants never mix with the main run.
