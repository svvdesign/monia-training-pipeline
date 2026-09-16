"""Unattended training loop for GitHub Actions.

Runs in a scheduled workflow (no laptop needed). Each invocation:
1. Downloads the current memoire.db + checkpoint (if any) from the
   Hugging Face Hub repo (persists state between runs, since GitHub
   Actions itself gives each run a totally fresh, empty machine).
2. Generates a modest batch of fresh Q&A pairs via the Gemini API,
   appended straight into memoire.db.
3. Trains for whatever's left of a fixed wall-clock budget (kept well
   under GitHub's hard 6-hour job limit), checkpointing to the Hub
   periodically mid-run so a hard timeout never loses more than ~20
   minutes of progress.
4. Uploads the final checkpoint + updated memoire.db back to the Hub.

Runs on GitHub's free hosted runners: 2 CPU cores, 7GB RAM, no GPU.
That RAM ceiling is why this uses the 0.5B model (not 1.5B) and plain
SGD (no momentum -- AdamW's extra per-parameter state would not fit).
"""
import os
import re
import sqlite3
import time

import requests
import torch
from huggingface_hub import hf_hub_download, upload_file
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- config from environment (set as GitHub Secrets / workflow env) -------
HF_TOKEN = os.environ["HF_TOKEN"]
HF_REPO = os.environ.get("HF_REPO", "saad3443/monia-checkpoint")
GEMINI_KEYS = [k.strip() for k in os.environ.get("GEMINI_KEYS", "").splitlines() if k.strip()]
GEMINI_CIBLE = int(os.environ.get("GEMINI_CIBLE", "500"))
TOTAL_BUDGET_S = int(os.environ.get("TOTAL_BUDGET_S", str(5 * 3600 + 30 * 60)))  # 5h30m, under the 6h hard cap
CHECKPOINT_PUSH_EVERY_S = 20 * 60  # push mid-run progress to the Hub every 20 min

DB_PATH = "/tmp/memoire.db"
CHECKPOINT_PATH = "/tmp/monia_entrainee.pth"
NOM_MODELE = "Qwen/Qwen2.5-0.5B-Instruct"
device = "cpu"

debut_script = time.time()


def temps_restant():
    return TOTAL_BUDGET_S - (time.time() - debut_script)


# --- step 1: pull current state from the Hub -------------------------------
print("Downloading memoire.db from the Hub...", flush=True)
try:
    p = hf_hub_download(HF_REPO, "memoire.db", token=HF_TOKEN)
    import shutil
    shutil.copy(p, DB_PATH)
    print(f"Got memoire.db ({os.path.getsize(DB_PATH)/1e6:.0f} MB).", flush=True)
except Exception as e:
    raise SystemExit(
        f"Could not download memoire.db from {HF_REPO}: {e}\n"
        "It needs to exist there before the first run -- upload it once manually."
    )

checkpoint_existe = False
try:
    p = hf_hub_download(HF_REPO, "monia_entrainee.pth", token=HF_TOKEN)
    import shutil
    shutil.copy(p, CHECKPOINT_PATH)
    checkpoint_existe = True
    print(f"Got existing checkpoint ({os.path.getsize(CHECKPOINT_PATH)/1e6:.0f} MB) -- resuming.", flush=True)
except Exception:
    print("No checkpoint on the Hub yet -- starting from the plain pretrained model.", flush=True)


def push_vers_hub(message):
    for chemin, nom in ((CHECKPOINT_PATH, "monia_entrainee.pth"), (DB_PATH, "memoire.db")):
        if os.path.exists(chemin):
            upload_file(
                path_or_fileobj=chemin, path_in_repo=nom, repo_id=HF_REPO,
                token=HF_TOKEN, commit_message=message,
            )
    print(f"Pushed to the Hub ({message}).", flush=True)


# --- step 2: grow the dataset a bit via Gemini ------------------------------
def charger_cles_gemini():
    return GEMINI_KEYS


SUJETS_GEMINI = [
    "world history", "science", "technology", "everyday practical advice",
    "cooking and food", "geography", "literature", "mathematics",
    "philosophy", "psychology", "law and civics", "business and economics",
    "sports", "music", "visual art", "computer programming", "cinema",
    "nature and wildlife", "space and astronomy", "health and medicine",
    "famous inventions", "world religions", "ancient civilizations",
    "climate and weather", "engineering", "linguistics", "architecture",
    "mythology", "critical thinking",
]
MODELE_GEMINI = "gemini-flash-lite-latest"
URL_GEMINI_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{modele}:generateContent?key={cle}"


def demander_lot_gemini(sujet, cle):
    import json
    prompt = (
        "Generate 25 diverse, high-quality question-and-answer pairs for training "
        f"a small local AI assistant, about the subject: {sujet}. Vary phrasing and "
        "question style. Each answer must be factually accurate, self-contained, and "
        "1-3 sentences. Return ONLY a JSON array, no markdown fences, no extra text: "
        '[{"question": "...", "answer": "..."}, ...]'
    )
    payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.95}}
    url = URL_GEMINI_TEMPLATE.format(modele=MODELE_GEMINI, cle=cle)
    try:
        r = requests.post(url, json=payload, timeout=60)
        data = r.json()
        if "error" in data:
            return [], data["error"].get("message", str(data["error"]))
        texte = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        texte = texte.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        paires_brutes = json.loads(texte)
        paires = [(p.get("question", "").strip(), p.get("answer", "").strip()) for p in paires_brutes]
        return [(q, a) for q, a in paires if q and a], None
    except Exception as e:
        return [], str(e)


if GEMINI_KEYS and GEMINI_CIBLE > 0:
    print(f"Generating up to {GEMINI_CIBLE} Gemini examples...", flush=True)
    conn = sqlite3.connect(DB_PATH)
    from datetime import datetime
    import random as _random
    inseres, appels, echecs = 0, 0, 0
    idx = 0
    while inseres < GEMINI_CIBLE and temps_restant() > TOTAL_BUDGET_S * 0.8:
        for cle in GEMINI_KEYS:
            if inseres >= GEMINI_CIBLE:
                break
            sujet = SUJETS_GEMINI[idx % len(SUJETS_GEMINI)]
            idx += 1
            paires, erreur = demander_lot_gemini(sujet, cle)
            appels += 1
            if erreur:
                echecs += 1
                time.sleep(2)
                continue
            if paires:
                conn.executemany(
                    "INSERT INTO dialogues (user_msg, ai_msg, created_at) VALUES (?, ?, ?)",
                    [(q, a, datetime.now().isoformat()) for q, a in paires],
                )
                conn.commit()
                inseres += len(paires)
            time.sleep(4.5)  # stay under free-tier per-key rate limits
    conn.close()
    print(f"Gemini: {inseres} examples added ({appels} calls, {echecs} failed).", flush=True)
else:
    print("Skipping Gemini generation (no keys or GEMINI_CIBLE=0).", flush=True)


# --- step 3: train for whatever's left of the budget ------------------------
MOTIF_TOUR_ASSISTANT = re.compile(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", re.S)
LONGUEUR_MAX_SEQUENCE = 256


def construire_labels_assistant_seulement(texte, tokenizer, max_length):
    encodage = tokenizer(texte, truncation=True, max_length=max_length, return_offsets_mapping=True)
    labels = [-100] * len(encodage["input_ids"])
    for m in MOTIF_TOUR_ASSISTANT.finditer(texte):
        for i, (d, f) in enumerate(encodage["offset_mapping"]):
            if d == f:
                continue
            if d >= m.start() and f <= m.end():
                labels[i] = encodage["input_ids"][i]
    return encodage["input_ids"], labels


def charger_paires(db_path, max_par_table=8000):
    conn = sqlite3.connect(db_path)
    paires = []
    for topic, content in conn.execute(
        "SELECT topic, content FROM knowledge WHERE source = 'enseigne-manuellement'"
    ).fetchall():
        phrases = re.split(r"(?<=[.!?])\s+", content.strip())
        paires.append((topic, phrases[0] if phrases else content[:200]))
    for topic, content in conn.execute(
        "SELECT topic, content FROM knowledge WHERE source != 'enseigne-manuellement' "
        "ORDER BY RANDOM() LIMIT ?", (max_par_table,)
    ).fetchall():
        phrases = re.split(r"(?<=[.!?])\s+", content.strip())
        paires.append((topic, phrases[0] if phrases else content[:200]))
    for u, a in conn.execute("SELECT user_msg, ai_msg FROM dialogues WHERE source = 'truthfulqa'").fetchall():
        paires.append((u, a))
    for u, a in conn.execute(
        "SELECT user_msg, ai_msg FROM dialogues WHERE source IS NULL OR source != 'truthfulqa' "
        "ORDER BY RANDOM() LIMIT ?", (max_par_table,)
    ).fetchall():
        paires.append((u, a))
    conn.close()
    return paires


SYSTEM_PROMPT_ENTRAINEMENT = (
    "You are Monia, Saad's local AI assistant. Use given context, stay on topic, be brief. "
    "If unsure of a fact, say so rather than guessing. Work multi-step questions through in order. "
    "Check your answer matches what was asked (e.g. 'how many' needs a number). "
    "Judge claims by evidence, not confidence. Separate correlation from causation. "
    "Acknowledge real nuance instead of flattening it."
)

print("Loading model...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(NOM_MODELE)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if checkpoint_existe:
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(NOM_MODELE)
    model = AutoModelForCausalLM.from_config(config).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
else:
    model = AutoModelForCausalLM.from_pretrained(NOM_MODELE).to(device)

print("Loading training pairs...", flush=True)
paires = charger_paires(DB_PATH)
print(f"{len(paires)} pairs loaded", flush=True)

import random
random.shuffle(paires)
textes = [
    tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT_ENTRAINEMENT},
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ],
        tokenize=False,
    )
    for q, a in paires
]

model.train()
# Plain SGD, no momentum: zero extra per-parameter state, since a
# GitHub-hosted runner only has 7GB RAM (AdamW's extra state would not fit
# even for the 0.5B model with real headroom to spare).
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
batch_size = 4
total_lots = (len(textes) + batch_size - 1) // batch_size
print(f"[TRAIN] {len(textes)} exemples, {total_lots} lots.", flush=True)

debut_entrainement = time.time()
dernier_push = time.time()
lots_faits = 0
for i in range(0, len(textes), batch_size):
    if temps_restant() <= 5 * 60:  # keep 5 min buffer to upload the final checkpoint
        print("Time budget nearly exhausted -- stopping training loop.", flush=True)
        break

    lot = textes[i:i + batch_size]
    paires_ids_labels = [construire_labels_assistant_seulement(t, tokenizer, LONGUEUR_MAX_SEQUENCE) for t in lot]
    paires_ids_labels = [(ids, lab) for ids, lab in paires_ids_labels if any(l != -100 for l in lab)]
    if not paires_ids_labels:
        continue
    longueur_max = max(len(ids) for ids, _ in paires_ids_labels)
    pad_id = tokenizer.pad_token_id
    input_ids, attention_mask, labels_batch = [], [], []
    for ids, labels in paires_ids_labels:
        manque = longueur_max - len(ids)
        input_ids.append(ids + [pad_id] * manque)
        attention_mask.append([1] * len(ids) + [0] * manque)
        labels_batch.append(labels + [-100] * manque)
    encodage = {
        "input_ids": torch.tensor(input_ids).to(device),
        "attention_mask": torch.tensor(attention_mask).to(device),
    }
    labels = torch.tensor(labels_batch).to(device)

    optimizer.zero_grad()
    perte = model(**encodage, labels=labels).loss
    if torch.isfinite(perte):
        perte.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    lots_faits += 1
    if lots_faits % 10 == 0:
        ecoule = time.time() - debut_entrainement
        print(f"[TRAIN] lot {lots_faits}/{total_lots} - perte {perte.item():.3f} - "
              f"{ecoule:.0f}s elapsed - {temps_restant():.0f}s left in budget", flush=True)

    if time.time() - dernier_push >= CHECKPOINT_PUSH_EVERY_S:
        model.eval()
        model.half()
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        model.float()
        model.train()
        push_vers_hub(f"mid-run checkpoint, {lots_faits} batches this run")
        dernier_push = time.time()

# --- step 4: final save + push ----------------------------------------------
model.eval()
model.half()
torch.save(model.state_dict(), CHECKPOINT_PATH)
push_vers_hub(f"run complete: {lots_faits} batches this run")
print("DONE", flush=True)
