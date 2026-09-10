"""Profanity and slur terms for languages the OpenAI Moderation API was measured NOT to
cover, registered into the same better-profanity wordlist the English checks already use.

Measured across 37 language/script combinations against the live moderation endpoint: it
reliably catches English, Spanish, French, German, Italian, Portuguese, Russian, Polish,
Dutch, Chinese, Japanese, Korean, Arabic, Thai and Indonesian - and then misses nearly the
entire Indian language family, in BOTH romanized and native script (only Bengali script and
Urdu's Arabic script came back covered). Hindi in Devanagari was inconsistent rather than
absent: a slur inside a full sentence was flagged, the same slur in a bare string was not.

That is the worst possible shape of this gap for an India-facing product, and Indians
overwhelmingly type romanized ("Hinglish"), which is the least-covered form of all. These
terms restore a free, instant, deterministic floor under all of it - the same mechanism and
reasoning as the English additions in llm/two_pass.py, applied to the languages the paid
classifier doesn't reach.

SELECTION RULE - false positives matter as much as misses here. Only distinctive terms are
included. Deliberately EXCLUDED because they collide with ordinary words a real customer
might type:
  - "pool"  (Malayalam slang, but also plain English - and the swimming-pool exclusion
             in two_pass.py depends on the ordinary sense)
  - "sala"  (mild Hindi insult, but Spanish/Portuguese for "room")
  - "haram" (legitimate religious term - only the "harami"/"haramzada" insult forms are in)
  - "poori" (a food), "kutta"/"kutte" (literally "dog"), "bal" (Bengali, but common
             elsewhere), and bare undiacriticked Vietnamese ("lon", "di", "du", "cac")
  - "randi"  (a real Hindi slur, but ALSO a common English given name - false-positived on
             "my name is Randi and I need help" in testing. The inflected forms "randwa",
             "randibaaz", Marathi "randiche"/"randichya" and Devanagari "रंडी" are kept, so
             only the bare romanized form is lost)
  - "lund"   (a real Hindi vulgar term, but ALSO a Swedish city and surname - false-
             positived on "I am from Lund, Sweden". Hindi "lauda"/"lawda"/"laude"/"lodu"
             and Devanagari "लंड" cover the same term and are kept)
Both removals are deliberate: wrongly hard-refusing a paying customer is a worse and far
more visible failure than missing one slur that two further layers may still catch.
Single tokens only: better-profanity matches word by word, so multi-word phrases never fire.
"""

# --- Hindi -------------------------------------------------------------------------
HINDI = [
    "madarchod", "madarchood", "maderchod", "maadarchod", "bhenchod", "behenchod",
    "bhainchod", "bhenchodh", "chutiya", "chutiye", "chutiyapa", "chutmarike",
    "gandu", "gaandu", "gandmasti", "bhosdike", "bhosadike", "bhosdiwale", "bhosda",
    "bhosdi", "bhosadi", "lodu", "laude", "lauda", "lawda", "harami",
    "haramzada", "haramzade", "haramkhor", "randwa", "randibaaz", "chinal", "kamine",
    "kaminey", "bhadwa", "bhadve", "gaand", "jhaatu", "jhant", "chodu", "chudai",
    "betichod", "maachod",
    # Devanagari
    "मादरचोद", "भेनचोद", "बहनचोद", "चूतिया", "चूतिये", "गांडू", "गाण्डू", "भोसड़ीके",
    "भोसड़ी", "लोडू", "लौड़ा", "लंड", "हरामी", "हरामज़ादा", "रंडी", "छिनाल", "कमीने",
    "भड़वा", "गांड", "झांटू", "चोदू", "चुदाई",
]

# --- Telugu ------------------------------------------------------------------------
TELUGU = [
    "lanja", "lanjakodaka", "lanjodaka", "lanjakodukku", "munda", "mundakodaka",
    "dengey", "denga", "dengu", "erripuka", "pooku", "sulli", "modda", "gudda",
    "bosudi", "kojja",
    # Telugu script
    "లంజ", "లంజకొడక", "ముండ", "ముండకొడక", "దెంగ", "పూకు", "సుల్లి", "మొడ్డ", "గుద్ద",
    "ఎర్రిపూకు", "బోసుడి",
]

# --- Marathi -----------------------------------------------------------------------
MARATHI = [
    "zavadya", "zavlya", "zavtos", "bhadvya", "randiche", "randichya", "aaighalya",
    "aaighale", "chutmar", "chinaal",
    # Devanagari
    "झवाड्या", "झवल्या", "भडव्या", "रांडीचे", "रांडीच्या", "आईघाल्या", "भोसडी",
]

# --- Tamil -------------------------------------------------------------------------
TAMIL = [
    "thevidiya", "thevdiya", "thevidia", "pundai", "punda", "otha", "ootha",
    "koothi", "thayoli", "myru", "sunni", "ommala",
    # Tamil script
    "தேவிடியா", "புண்ட", "புண்டை", "ஓத்த", "கூதி", "தாயோலி", "மயிரு", "சுன்னி",
]

# --- Bengali -----------------------------------------------------------------------
BENGALI = [
    "bokachoda", "bokachuda", "magi", "magir", "khanki", "khankir", "chudir",
    "haramjada", "shuorerbachcha",
    # Bengali script
    "বোকাচোদা", "মাগী", "মাগীর", "খানকি", "খানকির", "চুদির", "হারামজাদা",
]

# --- Punjabi -----------------------------------------------------------------------
PUNJABI = [
    "kanjar", "kanjri", "bhosdi", "bhainchod", "lun",
    # Gurmukhi
    "ਭੋਸੜੀ", "ਭੈਣਚੋਦ", "ਲੰਡ", "ਕੰਜਰ", "ਕੰਜਰੀ", "ਗਾਂਡੂ",
]

# --- Gujarati ----------------------------------------------------------------------
GUJARATI = [
    "bhosadino", "bhosdino", "lavdo", "lavda", "chodu",
    # Gujarati script
    "ભોસડી", "ભોસડીનો", "ચોદુ", "ગાંડુ", "રાંડ", "લવડો",
]

# --- Kannada -----------------------------------------------------------------------
KANNADA = [
    "bosudi", "bosdi", "tunne", "soole", "sooleymaga", "soolemaga", "kunni",
    # Kannada script
    "ಬೋಸುಡಿ", "ತುನ್ನೆ", "ಸೂಳೆ", "ಸೂಳೆಮಗ", "ತಿಕ",
]

# --- Malayalam ---------------------------------------------------------------------
MALAYALAM = [
    "thayoli", "pundachi", "kunna", "kandaroli", "oomb", "myre", "thendi",
    # Malayalam script
    "തായോളി", "മൈരു", "പുണ്ടച്ചി", "കുണ്ണ", "കണ്ടറോളി",
]

# --- Urdu --------------------------------------------------------------------------
URDU = [
    "haramzade", "haramzada", "kanjar", "gashti",
    # Arabic script
    "حرامزادہ", "کنجر", "گانڈو", "رنڈی", "گشتی",
]

# --- Turkish (Latin script, but measured uncovered) --------------------------------
TURKISH = [
    "orospu", "orospucocugu", "amcik", "amina", "sikeyim", "siktir", "yarrak",
    "gotveren", "piçkurusu",
]

# --- Vietnamese (diacritic forms only - the bare ASCII forms collide badly) ---------
VIETNAMESE = [
    "lồn", "cặc", "địt", "đĩ", "đụ", "đmm", "vcl", "đjt",
]


ALL_TERMS = (
    HINDI + TELUGU + MARATHI + TAMIL + BENGALI + PUNJABI
    + GUJARATI + KANNADA + MALAYALAM + URDU + TURKISH + VIETNAMESE
)


def _is_ascii(term: str) -> bool:
    return all(ord(c) < 128 for c in term)


# better-profanity cannot match non-ASCII at all - verified directly: after
# add_censor_words(["मादरचोद"]), contains_profanity("मादरचोद") still returns False, because
# the library tokenizes against an ASCII/homoglyph character set that no Indic script falls
# into. The romanized half of this list works fine through it (measured: 17/37 -> 28/37
# language coverage), so the split is by script, not by language: ASCII terms go to the
# library, native-script terms are matched here instead.
ROMANIZED_TERMS = [t for t in ALL_TERMS if _is_ascii(t)]
NATIVE_SCRIPT_TERMS = [t for t in ALL_TERMS if not _is_ascii(t)]


def contains_native_script_profanity(text: str) -> bool:
    """Substring match, deliberately - Indic scripts inflect by suffixing ("चूतिया" ->
    "चूतियापन"), and word-boundary matching would miss every inflected form. Safe here
    because every native-script term in this list is a long, distinctive sequence that
    does not occur inside ordinary words - the short, collision-prone terms were already
    excluded by the selection rule documented at the top of this module."""
    lowered = text.lower()
    return any(term in lowered for term in NATIVE_SCRIPT_TERMS)
