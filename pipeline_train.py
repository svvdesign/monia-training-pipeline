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
That RAM ceiling is why this uses LoRA (freezes the big pretrained model,
trains only a small adapter) instead of full fine-tuning -- full
fine-tuning of a 1.5B model needs ~12GB+ even with plain SGD, which does
not fit; LoRA on the frozen fp16 base needs only ~3-4GB. LoRA is also why
this can afford to use DeepSeek-R1-Distill-Qwen-1.5B (a model distilled
specifically to reason step-by-step, not just imitate answer tone) instead
of a plain instruction model -- full fine-tuning that model would not fit
here at all.
"""
import os
import re
import sqlite3
import time

import requests
import torch
from huggingface_hub import hf_hub_download, upload_file
from peft import LoraConfig, get_peft_model, PeftModel
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
NOM_MODELE = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
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

# Filename includes the model name deliberately -- switching base models
# (e.g. away from the earlier plain Qwen2.5-0.5B run) means a differently
# shaped state dict, so this avoids ever trying to load one architecture's
# weights into another instead of silently crashing on a shape mismatch.
CHECKPOINT_NOM_HUB = "monia_deepseek_r1_distill_1.5b_lora_merged.pth"

checkpoint_existe = False
try:
    p = hf_hub_download(HF_REPO, CHECKPOINT_NOM_HUB, token=HF_TOKEN)
    import shutil
    shutil.copy(p, CHECKPOINT_PATH)
    checkpoint_existe = True
    print(f"Got existing checkpoint ({os.path.getsize(CHECKPOINT_PATH)/1e6:.0f} MB) -- resuming.", flush=True)
except Exception:
    print("No checkpoint of this model on the Hub yet -- starting from the plain pretrained model.", flush=True)


def push_vers_hub(message):
    for chemin, nom in ((CHECKPOINT_PATH, CHECKPOINT_NOM_HUB), (DB_PATH, "memoire.db")):
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
LONGUEUR_MAX_SEQUENCE = 256


def construire_ids_et_labels(question, reponse, tokenizer, max_length):
    """Mask everything except the assistant's own response -- by TOKEN-LENGTH
    boundary, not by regex-matching a specific chat template's special
    tokens. Confirmed real bug this replaces: DeepSeek-R1-Distill-Qwen does
    not use Qwen's ChatML <|im_start|>/<|im_end|> markers, so a regex tuned
    for that format silently matched nothing -- every batch ended up with
    zero real labels and the whole training loop skipped every batch
    without doing any actual compute. Rendering the system+user prefix
    alone (with add_generation_prompt=True) and comparing its token length
    against the full system+user+assistant rendering works for ANY chat
    template, since it never inspects the template's actual tokens."""
    messages_prefixe = [
        {"role": "system", "content": SYSTEM_PROMPT_ENTRAINEMENT},
        {"role": "user", "content": question},
    ]
    prefixe_texte = tokenizer.apply_chat_template(messages_prefixe, tokenize=False, add_generation_prompt=True)
    texte_complet = tokenizer.apply_chat_template(
        messages_prefixe + [{"role": "assistant", "content": reponse}],
        tokenize=False, add_generation_prompt=False,
    )

    ids_prefixe = tokenizer(prefixe_texte, add_special_tokens=False)["input_ids"]
    ids_complet = tokenizer(
        texte_complet, truncation=True, max_length=max_length, add_special_tokens=False
    )["input_ids"]

    n_prefixe = min(len(ids_prefixe), len(ids_complet))
    labels = [-100] * n_prefixe + ids_complet[n_prefixe:]
    return ids_complet, labels


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
    # Resuming: the earlier run already merged its LoRA adapter into a
    # regular full state dict (see the save step below), so this loads
    # like any other checkpoint -- no PEFT wrapping needed here, only for
    # a *fresh* start (below), which needs to freeze the base weights.
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(NOM_MODELE)
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
else:
    # fp16 base weights (not fp32) is what makes this fit in 7GB -- frozen,
    # so precision here barely matters; only the small LoRA adapter (fp32,
    # for stable gradients) actually gets trained.
    base_model = AutoModelForCausalLM.from_pretrained(NOM_MODELE, torch_dtype=torch.float16).to(device)
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

print("Loading training pairs...", flush=True)
paires = charger_paires(DB_PATH)
print(f"{len(paires)} pairs loaded", flush=True)

import random
random.shuffle(paires)

model.train()
# AdamW is fine here (unlike the earlier full-fine-tune attempts) because
# only the small LoRA adapter has trainable parameters -- its extra
# per-parameter state is a few MB, not gigabytes.
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad], lr=2e-4
)


def sauvegarder_checkpoint():
    """Save a plain, resumable state dict with the LoRA adapter folded in.

    Confirmed real bug this replaces: merge_adapter() + get_base_model()
    does NOT flatten LoRA into plain .weight tensors in this PEFT version
    -- the target layers stay wrapped as .base_layer/.lora_A/.lora_B, so
    the saved state dict didn't match the plain architecture at all and
    the NEXT run's load_state_dict() failed with "Missing/Unexpected
    key(s)". merge_and_unload() actually flattens them, but it's
    destructive (replaces the live layers, no going back) -- so it runs on
    a deep copy, leaving the real training model untouched, and the copy
    is dropped right after saving.
    """
    if isinstance(model, PeftModel):
        import copy
        import gc
        copie = copy.deepcopy(model).merge_and_unload()
        copie.eval()
        etat_fp16 = {k: v.half() for k, v in copie.state_dict().items()}
        del copie
        gc.collect()
    else:
        model.eval()
        etat_fp16 = {k: v.half() for k, v in model.state_dict().items()}
        model.train()
    torch.save(etat_fp16, CHECKPOINT_PATH)


batch_size = 4
total_lots = (len(paires) + batch_size - 1) // batch_size
print(f"[TRAIN] {len(paires)} exemples, {total_lots} lots.", flush=True)

debut_entrainement = time.time()
dernier_push = time.time()
lots_faits = 0
for i in range(0, len(paires), batch_size):
    if temps_restant() <= 5 * 60:  # keep 5 min buffer to upload the final checkpoint
        print("Time budget nearly exhausted -- stopping training loop.", flush=True)
        break

    lot = paires[i:i + batch_size]
    paires_ids_labels = [construire_ids_et_labels(q, a, tokenizer, LONGUEUR_MAX_SEQUENCE) for q, a in lot]
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
        sauvegarder_checkpoint()
        push_vers_hub(f"mid-run checkpoint, {lots_faits} batches this run")
        dernier_push = time.time()

# --- step 4: final save + push ----------------------------------------------
sauvegarder_checkpoint()
push_vers_hub(f"run complete: {lots_faits} batches this run")
print("DONE", flush=True)
