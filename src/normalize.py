import re
import unicodedata

LEGAL = {
    "corporation": "corp",
    "incorporated": "inc",
    "company": "co",
    "limited": "ltd",
    "private": "pvt",
    "llc": "llc",
    "plc": "plc",
}

def normalize_text(value):
    if value is None:
        return ""
    s = str(value).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("&", " and ")
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_name(value):
    s = normalize_text(value)
    for a, b in LEGAL.items():
        s = re.sub(rf"\b{re.escape(a)}\b", b, s)
    return re.sub(r"\s+", " ", s).strip()

def normalize_address(value):
    s = normalize_text(value)
    replacements = {
        r"\bstreet\b": "st",
        r"\broad\b": "rd",
        r"\bavenue\b": "ave",
        r"\bdrive\b": "dr",
        r"\blane\b": "ln",
        r"\bhighway\b": "hwy",
        r"\bapartment\b": "apt",
        r"\bsuite\b": "ste",
    }
    for pattern, repl in replacements.items():
        s = re.sub(pattern, repl, s)
    return re.sub(r"\s+", " ", s).strip()

def compact(s):
    return re.sub(r"\W+", "", s or "")

def token_set(s):
    return {t for t in (s or "").split() if len(t) >= 2}

def prepare_frame(df):
    out = df.copy()
    out["name_norm"] = out["business_name"].map(normalize_name)
    out["address_norm"] = out["business_address"].map(normalize_address)
    out["name_compact"] = out["name_norm"].map(compact)
    out["address_compact"] = out["address_norm"].map(compact)
    out["name_prefix"] = out["name_compact"].str[:6]
    out["address_prefix"] = out["address_compact"].str[:8]
    out["name_tokens"] = out["name_norm"].map(token_set)
    out["address_tokens"] = out["address_norm"].map(token_set)
    return out
