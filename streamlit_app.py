import streamlit as st
from openai import OpenAI
import os
import re
from textwrap import shorten
from typing import Dict, List, Optional, Tuple

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
comp = complement_mrna(seq)

st.text_area("mRNA sequence (from dataset)", value=seq, height=140, disabled=True)
st.markdown("**Complement mRNA:**")
user_comp = st.text_area("", value=comp, height=140, key="comp_edit", disabled=False)

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
# ChatGPT prompt
# =============================
st.markdown("---")
st.subheader("Dataset-wide Similarity (ChatGPT)")

prompt_template = f"""
You are a bioinformatics assistant. I will provide a single mRNA sequence and its complementary mRNA (not reverse-complement).
Using the official miRBase dataset available at {MIRBASE_URL}, please:
1) Search for the closest matches to the given mRNA across the dataset.
2) Compute and report a 4-step comparison for the top match(es).
3) Also analyse the complementary mRNA if relevant.
Return a clean, well-structured HTML summary.

mRNA (from dataset record #{sel_recno}):\n{seq}\n\nComplement mRNA (editable):\n{user_comp}
"""

MODEL_NAME = "gpt-4o-mini"
ask = st.button("Ask ChatGPT now")

if ask:
    if not OPENAI_API_KEY:
        st.error("OpenAI API key is missing in secrets.")
    else:
        with st.status("Contacting ChatGPT…", expanded=False):
            try:
                from openai import OpenAI
                client = OpenAI(api_key=OPENAI_API_KEY)
                resp = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are a concise, precise bioinformatics assistant."},
                        {"role": "user", "content": prompt_template},
                    ],
                    temperature=0.2,
                )
                html_out = resp.choices[0].message.content if resp.choices else "(No response)"
                st.markdown("---")
                st.subheader("ChatGPT Response")
                st.markdown(html_out, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"OpenAI request failed: {e}")

st.markdown("---")
with st.expander("Preview first 10 records", expanded=False):
    preview = df_records[["RecordNo", "ID", "PrimaryAccession", "Length"]].head(10)
    st.dataframe(preview, use_container_width=True)

st.caption("Dataset is loaded directly from the miRBase URL. No local data file is needed.")



