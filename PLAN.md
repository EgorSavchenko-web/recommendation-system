# PLAN — RL fine-tuning of a GPT-2 sequential recommender on Yambda-50M

Status: APPROVED 2026-10-14
Deadline: 2026-10-20 (submission)
Task: T-Lab test task #2 (RL for sequential recommendation)
Compute: A100 via ClearML
Report language: English (README in a GitHub repo)

---

## 0. Scope and priorities

Grading criteria, in priority order:
1. A working, leakage-free experiment: base model, reward model (RM), RL model, all under the same evaluation protocol.
2. Depth of analysis: behaviour shift, the relevance/reward trade-off, several reward variants.
3. Clear conclusions and limitations.

Rule: criterion 1 must be finished by day 4. Everything else is layered on top.

---

## 1. Data (Yambda-50M)

- Source: `yandex/yambda` on Hugging Face, flat `multi_event` table (listens, likes, dislikes, unlikes, undislikes), plus album/artist mappings if available.
- Fields used: `uid, item_id, timestamp, event_type, is_organic, played_ratio_pct, track_length_seconds`.
- The meaning of the `is_organic` encoding must be verified against the dataset card or code before use, because the card is ambiguous.
- **Split (global temporal):**
  - test = last 7 days
  - val = the 7 days before test
  - train = everything earlier
  - Every model sees context strictly before the target event.
  - Ties at equal timestamps (5 s binning) are ordered deterministically and never leak forward.
- **Vocabulary:** top-N items by train-period frequency (N ≈ 50–100k; choose N after measuring coverage), plus `[OOV]`, `[PAD]`, `[BOS]`. Report the coverage of listens. Alternative if coverage is poor: multi-hash item embeddings, as in Matveev et al.
- **Sequences:** per-user chronological stream, chunked into windows of 512 events.
- **Token = event.** Input embedding = item emb + feedback emb (quantised ratio bucket, like flag) + organic flag emb + duration bucket emb. This uses the GPT-2 backbone via `inputs_embeds`.
- **Sessions:**
  - (a) Runs of consecutive non-organic events: the recommender-controlled trajectories, as in Matveev et al.
  - (b) Time-gap sessions (gap > 30 min). Used for the long-term reward.

## 2. Stage 1 — base model (next-item prediction)

- GPT-2 (Hugging Face `GPT2Model`), approx. 4 layers, 256 hidden, 4 heads. Start from the config in Matveev et al. and scale if time allows.
- Loss: full softmax CE over the vocabulary. This is affordable at N ≤ 100k on an A100.
- Metrics (val/test): Recall@10/100, NDCG@10/100, MAP@10.
- Two target sets: all events, and positive events (ratio ≥ threshold or like).
- Baselines: MostPop, plus the base GPT-2.

## 3. Stage 2 — reward model (user feedback model)

- Same backbone, initialised from the base model.
- Input: history plus candidate item.
- Heads: P(like), predicted listened ratio, and P(skip) as an optional extra.
- Default reward: `r = 1{like} + 0.1·ratio` (Matveev et al.). Alternatives are listed in section 5.
- Evaluation on held-out time: ratio MAE, like PR-AUC (report the base rate), calibration plot.
- **Ensemble of K = 3–5 RMs** with different seeds. Used for the uncertainty penalty and hacking detection.
- **"Gold" RM:** trained on a disjoint user half and never used for optimisation. It acts as an independent judge: if proxy reward rises while gold reward falls, that is evidence of reward hacking.
- Known limitation to state: selection bias. The RM only sees items users actually received, so it extrapolates on items the policy invents.

## 4. Stage 3 — RL fine-tuning

- **State:** a real history prefix from train sessions. **Action:** the next item from the vocabulary.
- **Main method:** on-policy policy gradient with a group baseline (RLOO / GRPO-style).
  - Sample G items per state.
  - Score each with the RM.
  - Advantage = reward − mean reward of the other samples in the group.
  - Loss = −A·log π + β·KL(π ‖ π_ref).
  - KL is computed **exactly** over the full vocabulary, which is cheap at this size.
- **Main knob:** β. Sweeping β gives the Pareto front of reward vs NDCG.
- **Multi-step variant:** roll out T = 5 items autoregressively. Each generated item is appended with RM-predicted feedback. Return = Σ γ^t r_t.
- **Optional, if time allows:** off-policy REINFORCE with importance weights on logged non-organic sessions (Chen et al. 2018; Matveev et al. 2026) as a comparison point.

## 5. Reward variants (one hypothesis each)

| ID | Reward | Expected failure mode |
|----|--------|-----------------------|
| R1 | listened ratio | shift to short tracks |
| R2 | listened seconds | shift to long tracks |
| R3 | like − dislike | popularity / familiar content |
| R4 | `1{like} + 0.1·ratio` (paper default) | reference |
| R5 | ratio relative to item/user baseline (advantage-style) | should resist the length bias |
| R6 | long-term proxy: P(session continues after this track) and/or return within N hours | tests the "long-term goal" question |
| +U | any reward − λ·std(ensemble) | pessimism to reduce RM exploitation |

## 6. Analysis (maps to the 5 research questions)

- **Q1, behaviour:** shifts in the distributions of track length, popularity (avg popularity, Gini, catalog coverage), repeat rate of already-heard tracks, artist concentration, and diversity via audio embeddings. Also proxy vs gold reward.
- **Q2, relevance and collapse:**
  - Recall/NDCG/MAP under the identical protocol.
  - Cross-user top-K Jaccard overlap.
  - Entropy of recommended items.
  - Share of users receiving the global top-10.
- **Q3, long-term:** R6 results, plus a written section on what is offline-testable versus what needs an A/B test. There are no logged propensities, so standard IPS is not available.
- **Q4, multi-step:**
  - Compare reward and relevance at step 1…5 when the context is real history versus generated history.
  - Measure drift: embedding distance, KL to the real-history next-item distribution, and RM ensemble disagreement as the context becomes synthetic.
- **Q5, trade-off:** Pareto curves over β, for each reward variant.

## 7. Repository layout (to create after approval)

```
README.md            # report (English)
configs/             # yaml per experiment
src/data/            # download, preprocessing, splits, vocab, sessions
src/models/          # gpt2 policy, reward model
src/train/           # train_base.py, train_rm.py, train_rl.py
src/eval/            # metrics.py, behaviour.py, multistep.py
scripts/             # ClearML task launchers
tests/               # pytest: split/leakage, metrics, KL, advantages
reports/figures/
```

## 8. Schedule

| Day | Date | Deliverable |
|-----|------|-------------|
| 1 | Oct 14–15 | Data pipeline, split, vocab coverage, leakage tests |
| 2 | Oct 16 | Base GPT-2 trained and evaluated |
| 3 | Oct 17 | RM ensemble + gold RM, evaluated |
| 4 | Oct 18 | RL (one-step + multi-step), β sweep for R4 → **criterion 1 done** |
| 5 | Oct 19 | Reward variants R1–R6 (+U), behaviour and multi-step analysis |
| 6 | Oct 20 | README report, figures, reproducibility check, submit |

## 9. References

The task rules require every cited fact to be human-verified. Each entry needs a tick before it goes into the README.

| Ref | Status |
|-----|--------|
| Petrov & Macdonald, *Aligning GPTRec with Beyond-Accuracy Goals with RL*, arXiv 2403.04875 | read (abstract) |
| Matveev et al., *Session-Level Optimization for Large-Scale Retrieval using REINFORCE with Multi-Step Off-Policy Correction*, arXiv 2607.02818 | read (full) |
| Ploshkin et al., *Yambda-5B*, arXiv 2505.22238 | to verify |
| Chen et al., *Top-K Off-Policy Correction for a REINFORCE Recommender System*, arXiv 1812.02353 | to verify (cited in Matveev et al.) |
| Gao et al., *Scaling Laws for Reward Model Overoptimization* (2022) | to verify |
| Shao et al., DeepSeekMath (GRPO) (2024) | to verify |
| Ahmadian et al., *Back to Basics: REINFORCE-style optimization for RLHF* (RLOO) (2024) | to verify |
| Radford et al., GPT-2 (2019) | to verify |

## 10. Infrastructure (decided)

- **Remote execution** follows the existing lab pattern:
  - `--clearml` + `--queue <name>` flags.
  - The queue is chosen per run; whichever is free gets the job.
  - `Task.init` → argv transferred as a task parameter → `set_packages` → `execute_remotely`.
  - Data comes from a ClearML Dataset via `Dataset.get(...).get_local_copy()`.
  - Outputs are uploaded as artifacts.
- **No infrastructure identifiers in source.** Project, default queue and dataset ids are read from env vars (`YAMBDA_CLEARML_PROJECT`, `YAMBDA_CLEARML_QUEUE`, `YAMBDA_DATASET_ID`, `YAMBDA_ARTIFACTS_DATASET_ID`).
- **Data ingest:** a one-off local script downloads from Hugging Face `flat/50m/multi_event.parquet` (~384 MB), `artist_item_mapping.parquet` and `album_item_mapping.parquet`, then registers them as a ClearML Dataset. Audio embeddings (13.8 GB) are skipped; diversity is measured via artist/album instead.
- **Model chaining:** base → RM → RL. Checkpoints are passed between stages as ClearML Datasets or task artifacts, referenced by id.
- **`is_organic`:** 1 = organic, 0 = recommendation-driven. Verified on the dataset card (fields table and FAQ agree).
- **Code location:** a local folder linked to a public GitHub repo. Commits are atomic; the user pushes.

## 10a. Data facts (measured 2026-10-14)

- 47.8M events: 46.5M listens, 881k likes, 313k unlikes, 108k dislikes, 21k undislikes.
- 10k users, 877k distinct listened items. The span is about 1505 days.
- 51.7% of listens are organic.
- Last-7-day test: 333k listens from 5.5k users. Val (the previous 7 days): 318k listens.
- `played_ratio_pct`: median 100, 25th percentile 7, so the distribution is strongly bimodal (skip vs full listen). 63% of listens are ≥ 50%. Median track length is 200 s.
- Likes per listen: 1.9%.
- **Vocabulary coverage** (top-N by train frequency → share of test listens covered):

  | Top N | Test coverage | Min train count |
  |-------|---------------|-----------------|
  | 20k | 60% | — |
  | 50k | 76% | — |
  | 100k | 85% | 50 |
  | 200k | 92% | 15 |

- **Decision:** N = 100k. CE over the full vocabulary is computed in token chunks to bound memory.

## 11. Open questions

- Can the ClearML agent clone from GitHub? This decides between a multi-module package and self-contained entry scripts.
