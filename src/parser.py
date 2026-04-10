"""
parser.py
---------
Stage 1 – Parsing functions.
Converts raw unstructured recipe text into a structured dict.

This is the first and most complex stage of the pipeline. It handles a wide
variety of recipe text formats: bullet-point lists, numbered steps, inline
"Ingredients: ..." lines, paragraph blocks, and more.
"""

from __future__ import annotations
import re
from typing import Any

from .config import (
    UNIT_MAP, INGREDIENT_NORM, INGR_PAT,
    RX_TIME, RX_TIME_H_ONLY, RX_TIME_MIN_ONLY, RX_SERVINGS,
    INGREDIENT_HEADERS, INSTRUCTION_HEADERS,
)

# ---------------------------------------------------------------------------
# Optional spaCy NLP integration
# spaCy is used for two purposes:
#   1. Splitting paragraph-form instructions into individual sentences
#      when no numbered/bulleted steps are detected.
#   2. Further splitting any single step that is very long (>100 chars).
# If spaCy is not installed or the model is missing, the pipeline falls back
# cleanly to regex-based sentence splitting.
# ---------------------------------------------------------------------------
NLP_AVAILABLE = False
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")  # small English model for sentence segmentation
        NLP_AVAILABLE = True
    except OSError:
        # Model not downloaded — run: python -m spacy download en_core_web_sm
        pass
except ImportError:
    # spaCy package not installed at all — regex fallback will be used
    pass


# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------

def normalize_unit(unit: str | None) -> str | None:
    """Return canonical unit or original if unknown.

    Looks up the unit in UNIT_MAP, first as-is (lowercased), then with a
    trailing 's' stripped to handle plural forms not explicitly in the map.
    Returns the original string (lowercased) if no match is found.
    """
    if not unit:
        return None
    key = unit.lower().strip().rstrip("s")   # strip trailing 's' for plural fallback
    return UNIT_MAP.get(unit.lower().strip(), UNIT_MAP.get(key, unit.lower().strip()))


def normalize_ingredient(item: str) -> str:
    """Map common ingredient aliases to canonical names.

    Strips parenthetical qualifiers (e.g. "(optional)") and comma-separated
    notes (e.g. "bread, sliced") before looking up in INGREDIENT_NORM.
    Returns the canonical name, or the cleaned original if no alias exists.
    """
    if not item:
        return item
    lower = item.lower().strip()
    lower = re.sub(r",.*$", "", lower).strip()      # remove ", sliced" / ", to taste" etc.
    lower = re.sub(r"\(.*?\)", "", lower).strip()   # remove "(optional)" / "(about 200g)" etc.
    return INGREDIENT_NORM.get(lower, lower)


# ---------------------------------------------------------------------------
# Quantity parsing
# ---------------------------------------------------------------------------

def parse_quantity(raw: str | None) -> float | None:
    """Convert fraction / range strings to float.

    Handles three formats:
        '3/4'  → 0.75          (fraction)
        '9-11' → 10.0          (range: returns midpoint)
        '2'    → 2.0           (plain integer or decimal)

    Returns None for unrecognisable strings or division-by-zero cases.
    """
    if raw is None:
        return None
    raw = raw.strip()

    # Range format: e.g. "9-11" or "2–3" (en-dash included via \u2013)
    range_m = re.match(r"(\d+(?:\.\d+)?)\s*[-\u2013]\s*(\d+(?:\.\d+)?)", raw)
    if range_m:
        # Use the midpoint of the range as a numeric estimate
        return round((float(range_m.group(1)) + float(range_m.group(2))) / 2, 3)

    # Fraction format: e.g. "3/4"
    if "/" in raw:
        parts = raw.split("/")
        try:
            return round(float(parts[0]) / float(parts[1]), 3)
        except (ValueError, ZeroDivisionError):
            return None   # malformed fraction — treat as missing

    # Plain numeric string
    try:
        return float(raw)
    except ValueError:
        return None   # unrecognisable format


# ---------------------------------------------------------------------------
# Ingredient line parser
# ---------------------------------------------------------------------------

def parse_ingredient(line: str) -> dict[str, Any]:
    """Parse one ingredient line into a structured dict.

    Uses the compiled INGR_PAT regex to extract quantity, unit, and item name.
    If the regex does not match (e.g. ingredient is just a name with no quantity),
    a fallback dict is returned with quantity and unit set to None.

    Returns keys:
        quantity        – numeric float or None
        unit            – canonical unit string or None
        item            – raw item name as found in the text
        normalized_item – canonical name from INGREDIENT_NORM, or None if already canonical
        original_text   – the full unstripped line for audit purposes
    """
    original = line   # preserve original before any modification for the audit trail
    line = line.strip(" -*\u2022\t")   # strip bullet characters and leading whitespace

    match = INGR_PAT.match(line)
    if match:
        # Regex matched — extract named capture groups
        qty_raw = match.group("qty")
        unit    = match.group("unit")
        item    = match.group("item").strip(" ,.")
        qty_val = parse_quantity(qty_raw)
        norm    = normalize_ingredient(item)
        return {
            "quantity":        qty_val,
            "unit":            normalize_unit(unit),
            "item":            item,
            # Only set normalized_item if it differs from the raw item (avoid redundancy)
            "normalized_item": norm if norm != item.lower().strip() else None,
            "original_text":   original.strip(),
        }

    # Fallback: regex did not match — return just the cleaned item name with no qty/unit
    norm = normalize_ingredient(line)
    return {
        "quantity":        None,
        "unit":            None,
        "item":            line,
        "normalized_item": norm if norm != line.lower().strip() else None,
        "original_text":   original.strip(),
    }


# ---------------------------------------------------------------------------
# Time extraction
# ---------------------------------------------------------------------------

def extract_time_minutes(text: str) -> int | None:
    """Extract prep/cooking time from a text line. Returns minutes or None.

    Tries three regex patterns in order of specificity:
        1. RX_TIME:         "1 hr 30 min" (hours + minutes combined)
        2. RX_TIME_H_ONLY:  "2 hours"     (hours only)
        3. RX_TIME_MIN_ONLY:"45 minutes"  (minutes only)

    Returns the total time in minutes as an integer, or None if no match.
    """
    m = RX_TIME.search(text)
    if m:
        hours = int(m.group("hours") or 0)
        mins  = int(m.group("mins")  or 0)
        total = hours * 60 + mins
        if total > 0:   # guard against a spurious "0 minutes" match
            return total

    # Try hours-only fallback (e.g. "Prep time: 2 hours")
    m = RX_TIME_H_ONLY.search(text)
    if m:
        return int(m.group("hours")) * 60

    # Try minutes-only fallback (e.g. "Time: 45 min")
    m = RX_TIME_MIN_ONLY.search(text)
    if m:
        return int(m.group("mins"))

    return None   # no time information found in this line


# ---------------------------------------------------------------------------
# Full recipe text → structured dict
# ---------------------------------------------------------------------------

def extract_recipe_features(text: str, recipe_id: str = "unknown") -> dict[str, Any]:
    """Parse raw recipe text into a structured recipe dict.

    Handles: bullet lists, numbered steps, inline sections, paragraph blocks.
    Falls back to spaCy sentence splitting when instructions are paragraph-form.

    The parser is stateful: it tracks a `section` variable that changes as
    section headers are encountered (preamble → ingredients → instructions).
    Lines within each section are processed differently depending on the current
    section context.

    Args:
        text:      Raw recipe text (may include newlines and bullet characters).
        recipe_id: Identifier assigned to the recipe (from source data).

    Returns:
        Structured recipe dict with keys:
        recipe_id, name, ingredients, instructions,
        prep_time_minutes, servings, tags, _meta
    """
    warnings: list[str] = []   # collects non-fatal parse issues for the validation report

    # Initialise the output dict with safe defaults for all expected fields
    data: dict[str, Any] = {
        "recipe_id":         recipe_id,
        "name":              None,
        "ingredients":       [],
        "instructions":      [],
        "prep_time_minutes": None,
        "servings":          None,
        "tags":              [],
        "_meta": {
            "imputed_fields":  [],    # filled in by imputation.py
            "original_values": {},    # pre-imputation values preserved here
            "parse_warnings":  warnings,   # reference — warnings list is shared
            "is_duplicate":    False,       # set by enrichment.py
            "duplicate_of":    None,        # recipe_id of the earlier duplicate
        },
    }

    # Split text into non-empty lines and strip surrounding whitespace
    lines = [L.strip() for L in text.split("\n") if L.strip()]
    if not lines:
        warnings.append("Empty text – no data extracted.")
        return data

    # First non-empty line is treated as the recipe name
    data["name"] = lines[0]
    section      = "preamble"   # tracks current section: preamble / ingredients / instructions
    instructions_raw: list[str] = []   # accumulates raw instruction strings before post-processing

    for line in lines[1:]:
        lower = line.lower()

        # ── Section header detection ───────────────────────────────────────
        # Strip trailing punctuation/whitespace from line before comparing
        # to the header keyword sets (e.g. "Ingredients:" → "ingredients")
        stripped_lower = re.sub(r"[:\-\s]+$", "", lower).strip()
        if stripped_lower in INGREDIENT_HEADERS:
            section = "ingredients"
            continue   # this line is only a header, not content
        if stripped_lower in INSTRUCTION_HEADERS:
            section = "instructions"
            continue

        # ── Inline "Instructions: first step here" ────────────────────────
        # Some recipes put the header and first instruction on the same line
        inline_instr = re.match(
            r"^(?:instructions?|steps?|directions?|method)\s*:\s*(.+)$", line, re.IGNORECASE
        )
        if inline_instr:
            section = "instructions"
            instructions_raw.append(inline_instr.group(1).strip())
            continue

        # ── Inline "Ingredients: a, b, c" ────────────────────────────────
        # Comma-separated ingredient list on a single line
        inline_ingr = re.match(r"^ingredients?\s*:\s*(.+)$", line, re.IGNORECASE)
        if inline_ingr:
            section = "ingredients"
            for part in inline_ingr.group(1).split(","):
                part = part.strip(" .")
                if part:
                    data["ingredients"].append(parse_ingredient(part))
            continue

        # ── Time extraction ───────────────────────────────────────────────
        # Only extract the first time value encountered; ignore subsequent ones
        time_val = extract_time_minutes(lower)
        if time_val is not None and data["prep_time_minutes"] is None:
            data["prep_time_minutes"] = time_val
            continue   # this line was a time annotation, not an ingredient/step

        # ── Servings extraction ───────────────────────────────────────────
        # Only extract the first servings value encountered
        svc_m = RX_SERVINGS.search(lower)
        if svc_m and data["servings"] is None:
            data["servings"] = int(svc_m.group("n"))
            continue   # this line was a servings annotation

        # ── Section content handling ──────────────────────────────────────
        if section == "ingredients":
            # Avoid accidentally capturing time or servings lines as ingredients
            if not re.search(r"time|serving|yield", lower):
                data["ingredients"].append(parse_ingredient(line))

        elif section == "instructions":
            # Strip common step prefixes: "Step 1:", "1.", "1)", "1-"
            clean = re.sub(r"^(?:step\s*)?\d+[.):\-]\s*", "", line, flags=re.IGNORECASE).strip()
            if clean and len(clean) >= 5:   # ignore very short fragments
                instructions_raw.append(clean)

        elif section == "preamble":
            # We're still in the preamble (before any section header).
            # Warn if a long line appears that might actually be instructions
            # but wasn't caught because no section header was present.
            if len(line) > 60 and not data["ingredients"]:
                warnings.append(f"Preamble line may contain instructions: '{line[:60]}...'")

    # ── spaCy fallback for paragraph-form instructions ────────────────────
    # If no discrete steps were found but we ended up in the instructions
    # section, try splitting the entire remaining text into sentences via NLP.
    if NLP_AVAILABLE and not instructions_raw and section == "instructions":
        full_para = " ".join(lines[1:])
        doc = nlp(full_para)
        instructions_raw = [s.text.strip() for s in doc.sents if len(s.text.strip()) > 10]
        if instructions_raw:
            warnings.append("Instructions extracted via NLP sentence splitting.")

    # ── Post-process: split long steps into shorter sentences ─────────────
    # A step longer than 100 characters may actually contain multiple actions.
    # Use spaCy if available, otherwise split on sentence-ending punctuation.
    final_instructions: list[str] = []
    for step in instructions_raw:
        if NLP_AVAILABLE and len(step) > 100:
            # spaCy segmentation gives more accurate sentence boundaries
            doc = nlp(step)
            for sent in doc.sents:
                s = sent.text.strip()
                if s and len(s) >= 5:
                    final_instructions.append(s)
        else:
            # Regex split on sentence-ending punctuation followed by a capital letter
            subs = re.split(r"(?<=[.!?])\s+(?=[A-Z])", step)
            final_instructions.extend(s.strip() for s in subs if s.strip() and len(s.strip()) >= 5)

    data["instructions"] = final_instructions

    # ── Final parse-level warnings ────────────────────────────────────────
    # These warnings feed into the validation report (check_type=parse_warning)
    if not data["ingredients"]:
        warnings.append("No ingredients extracted.")
    if not data["instructions"]:
        warnings.append("No instructions extracted.")
    if data["prep_time_minutes"] is None:
        warnings.append("Prep time not found in text.")
    if data["servings"] is None:
        warnings.append("Servings not found in text.")

    return data
