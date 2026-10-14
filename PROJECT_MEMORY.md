# PROJECT_MEMORY

## Context
- Purpose: T-Bank T-Lab internship selection, test task #2. One of two tasks may be chosen; #2 was chosen.
- Deadline: 2026-10-20.
- Rules from the organisers:
  - Reasonable AI use is allowed.
  - Every fact and citation in the final text must be human-verified (existence of the paper and the claim). Violations may annul the work.
  - Questions go to the Telegram bot @clar_tbot (Академия → «Вопрос по ML») or to best-talents@tbank.ru.

## Requirements (from the task text)
1. Train a GPT-2-based next-item model; item IDs are tokens. Dataset: Yambda-50M (chosen).
2. Train a reward model predicting the user response (listened ratio / like) given history; evaluate on held-out data.
3. Fine-tune with RL toward higher expected reward, scored by the RM.
4. Analyse:
   - behaviour shift
   - relevance (Recall/NDCG/MAP) and collapse to the same items for everyone
   - long-term reward design (offline vs online)
   - multi-step (5-item) generation drift
   - the relevance/reward trade-off
5. Time-aware split with no leakage, and the same evaluation protocol for all models.
6. A negative RL result with a good explanation is acceptable.
7. Deliverable: GitHub repo with README report (English), code, and reproduction instructions.

## Decisions
- Dataset: Yambda-50M.
- Compute: A100 via the lab's ClearML. The supervisor has approved use.
- Report: English, GitHub README.
- Closest prior work: Matveev et al. 2026 (arXiv 2607.02818). Same dataset family, off-policy REINFORCE, user feedback model with like + ratio heads, reward = 1{like} + 0.1·ratio, last-week test split, 4 layers / 256 dim / 4 heads.

## Known pitfalls
- `is_organic`: 1 = organic, 0 = recommendation-driven (verified on the dataset card).
- ClearML: the queue varies per run (`--queue` flag). No infrastructure ids in source; they come from env vars.
- Timestamps are binned to 5 s, so ties must be ordered deterministically.
- The catalog has about 934k items, so the vocabulary must be truncated (or hashed). Report coverage.
- RM selection bias leads to reward hacking on rarely-shown items.
