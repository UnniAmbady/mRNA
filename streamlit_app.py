from openai import OpenAI
import re
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st
from difflib import SequenceMatcher

# =============================
# App Title
# =============================
st.set_page_config(page_title="mRNA Complement Analyser", layout="wide")
st.title("mRNA Complement Analyser")

# =============================
# Secrets and constants
# =============================
OPENAI_API_KEY = st.secrets.get("openai", {}).get("secret_key", "")
MIRBASE_URL = "https://www.mirbase.org/download/miRNA.dat"

# =============================
# Utilities – parsing EMBL-like miRNA.dat
# =============================

def load_text_from_url(url: str) -> str:
    import urllib.request
    with urllib.request.urlopen(url) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def split_entries(text: str) -> List[str]:
    return [c.strip() for c in text.strip().split("//") if c.strip()]


def unfold_multilines(lines: List[str], key: str) -> List[str]:
    return [ln for ln in lines if ln.startswith(key)]


def parse_accessions(lines: List[str]) -> List[str]:
    acc_lines = unfold_multilines(lines, "AC")
    accs: List[str] = []
    for ln in acc_lines:
        rest = ln[2:].strip()
        parts = [p.strip() for p in rest.split(";") if p.strip()]
        accs.extend(parts)
    # Preserve order / uniqueness
    seen = set()
    out = []
    for a in accs:
        if a not in seen:
            out.append(a)
            seen.add(a)
    return out


def parse_id_block(first_line: str) -> Dict[str, Optional[str]]:
    parts = first_line.strip().split()
    id_name = parts[1] if len(parts) > 1 else None
    m = re.search(r";\s*([A-Za-z]+);\s*([A-Za-z]+);\s*([A-Za-z]+);\s*(\d+)\s*BP\.", first_line)
    length = int(m.group(4)) if m else None
    taxon = m.group(3) if m else None
    mol_type = m.group(2) if m else None
    return {"ID": id_name, "Length": length, "TaxonCode": taxon, "MolType": mol_type}


def parse_sequence(lines: List[str]) -> str:
    seq = []
    in_seq = False
    for ln in lines:
        if ln.startswith("SQ"):
            in_seq = True
            continue
        if in_seq:
            if ln[:2] in {"XX", "FH", "FT", "RN", "ID", "DE", "AC"}:
                in_seq = False
                continue
            seq.append(re.sub(r"[^ACGTUNacgtun]", "", ln))
    return "".join(seq).lower()


def parse_description(lines: List[str]) -> str:
    d_lines = unfold_multilines(lines, "DE")
    desc = " ".join(ln[2:].strip() for ln in d_lines).strip()
    return re.sub(r"\s+", " ", desc)


def parse_entry(entry_text: str) -> Dict[str, object]:
    lines = [ln.rstrip("\n") for ln in entry_text.splitlines() if ln.strip()]
    if not lines:
        return {}
    id_info = parse_id_block(lines[0])
    accs = parse_accessions(lines)
    desc = parse_description(lines)
    seq = parse_sequence(lines)
    return {
        **id_info,
        "Accessions": accs,
        "PrimaryAccession": accs[0] if accs else None,
        "Description": desc,
        "Sequence": seq,
    }


def build_dataframe(entries: List[Dict[str, object]]) -> pd.DataFrame:
    recs = []
    for idx, e in enumerate(entries, start=1):
        if not e:
            continue
        recs.append({
            "RecordNo": idx,
            "ID": e.get("ID"),
            "PrimaryAccession": e.get("PrimaryAccession"),
            "Description": e.get("Description"),
            "Length": e.get("Length"),
            "Sequence": e.get("Sequence"),
        })
    return pd.DataFrame(recs)


def complement_mrna(seq: str) -> str:
    # mRNA complement (not reverse-complement): A<->U, C<->G; tolerate T as U.
    comp_map = str.maketrans({
        "a": "u", "u": "a", "g": "c", "c": "g",
        "t": "a",
        "n": "n",
        "A": "U", "U": "A", "G": "C", "C": "G", "T": "A", "N": "N",
    })
    return (seq or "").translate(comp_map)


def gc_content(seq: str) -> float:
    if not seq:
        return 0.0
    s = seq.lower()
    gc = sum(1 for ch in s if ch in "gc")
    return round(100.0 * gc / len(s), 2)


def similar_ratio(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return round(100.0 * SequenceMatcher(None, a, b).ratio(), 2)

@st.cache_data(show_spinner=False)
def parse_dataset() -> pd.DataFrame:
    raw = load_text_from_url(MIRBASE_URL)
    entries = [parse_entry(et) for et in split_entries(raw)]
    df = build_dataframe(entries)
    return df

# =============================
# Load dataset
# =============================
try:
    df_records = parse_dataset()
    max_records = int(df_records.shape[0])
except Exception as e:
    st.error(f"Could not load dataset: {e}")
    st.stop()

st.success(f"Dataset loaded from URL. Maximum number of records: {max_records}")

# =============================
# Main UI – record slider and fields
# =============================
sel_recno = st.slider("Recorde: ", min_value=1, max_value=max_records, value=1, step=1)
st.markdown(f"**Recorde:** #{sel_recno}")

row = df_records.loc[df_records["RecordNo"] == sel_recno].iloc[0]
seq = row.get("Sequence", "")
comp_seq = complement_mrna(seq)

st.text_area("mRNA sequence (from dataset)", value=seq, height=140, disabled=True)
st.markdown("**Complement mRNA:**")
user_comp = st.text_area("", value=comp_seq, height=140, key="comp_edit", disabled=False)

# Quick local stats (kept for convenience)
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Length (mRNA)", f"{len(seq)} nt")
with col2:
    st.metric("Length (Complement)", f"{len(user_comp)} nt")
with col3:
    st.metric("GC% (mRNA)", f"{gc_content(seq)}%")
with col4:
    st.metric("Similarity (SeqMatcher)", f"{similar_ratio(seq, user_comp)}%")

# =============================
# A) Ask Chat GPT Now – build required prompt and send
# =============================
st.markdown("---")
st.subheader("Dataset-wide Similarity (Chat GPT)")

button = st.button("Ask Chat GPT Now")

if button:
    if not OPENAI_API_KEY:
        st.error("OpenAI API key is missing in secrets.")
    else:
        # Build the EXACT prompt shape requested by the user
        prompt_text = f"""
Data Source = "{MIRBASE_URL}"
Complement mRNA = {user_comp}
compare the above with entire data base and find the Similarities and highlight the reord numers /names.
The comparison must include:
What are the entities to be computed?
1. Basic checks
   • Lengths, %GC, presence of AUG start and proper stop (UAA/UAG/UGA).
2. Similarity
   • Global alignment (best for full-length mRNAs): percent identity, gaps.
   • Edit distance (minimum number of changes).
3. Change list (“what to fix”)
   • A position-by-position list of substitutions, insertions, deletions with 1-based coordinates (e.g., c.125A>G, c.300_301insU, c.451delC).
4. Codon-aware impact (optional but recommended)
   • Translate both and report synonymous / missense / nonsense / frameshift changes with protein effects (e.g., p.Glu42Lys, p.Trp85Ter).
   • Flag any indels not divisible by 3 (frameshifts).
------------------------------------------------------------
Give a compleste Comaparison Analysis to at least top 2 similar mRNA.
"""
        
        with st.status("Contacting Chat GPT…", expanded=False):
            try:
                from openai import OpenAI
                client = OpenAI(api_key=OPENAI_API_KEY)
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": (
                            "You are a rigorous bioinformatics assistant. Return PLAIN TEXT or MARKDOWN only. Do NOT include HTML tags or DOCTYPE. Structure with headings, lists, and code blocks if helpful."
                        )},
                        {"role": "user", "content": prompt_text},
                    ],
                    temperature=0.1,
                )
                resp_text = resp.choices[0].message.content if resp.choices else "(No response)"

                st.markdown("---")
                st.subheader("Chat GPT Response")
                st.markdown(resp_text)
            except Exception as e:
                st.error(f"OpenAI request failed: {e}")

# C) Removed the previous preview table per request.

st.caption("Dataset is loaded directly from the miRBase URL. No local data file is needed.")





