# Monia unattended training pipeline

Runs Monia's training loop automatically every 6 hours on GitHub's free
hosted runners — no laptop or paid server needed. Each run:

1. Downloads the current `memoire.db` + checkpoint from the Hugging Face
   Hub repo (`HF_REPO`), since GitHub Actions gives every run a fresh,
   empty machine.
2. Generates a batch of new Q&A pairs via the Gemini API.
3. Trains the model (Qwen2.5-0.5B — see below for why not 1.5B) for up to
   ~5h30m, checkpointing back to the Hub every 20 minutes mid-run.
4. Uploads the final state back to the Hub for the next run to resume from.

## Why the 0.5B model, not 1.5B

GitHub's free runners have only 7GB RAM and no GPU. That's not enough to
fit the 1.5B model even with memory-lean SGD. This pipeline trades model
size for the ability to run completely unattended and free — run the 1.5B
model manually on Kaggle when convenient; the two aren't exclusive.

## One-time setup (already done by Claude, documented here for reference)

- `HF_TOKEN` and `GEMINI_KEYS` are stored as GitHub Secrets (Settings →
  Secrets and variables → Actions) — never committed to this repo.
- The initial `memoire.db` was uploaded directly to the Hugging Face repo
  before the first scheduled run, since the pipeline itself only *updates*
  what's already there.

## Checking on it

- **Actions tab** on this repo shows every run, live logs, and lets you
  trigger one manually ("Run workflow").
- The current checkpoint is always at
  `https://huggingface.co/svvdesign/monia-checkpoint`.
- To pull the latest checkpoint down to your Mac, download
  `monia_entrainee.pth` from that Hugging Face repo's Files tab.
