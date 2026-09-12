"""One focused decision: can Geometra measure the thing the customer is asking about?

WHY THIS EXISTS, AND WHY IT IS NOT ANOTHER WORDLIST

Rule 8/8B/8C is the capability sheet, marked SUPREME AUTHORITY in the Pass 2 prompt. It is
also 9,600 characters inside a 22,623-character prompt, competing with tone, clarification
policy, ticket policy, safety rules and source formatting - while the same call is also
asked to WRITE the answer. Measured, Pass 2 does not apply it reliably:

  - "I want to measure lasagna"  -> hard SAFETY refusal (food is named in Rule 8C)
  - "can I measure biryani"      -> hard SAFETY refusal
  - "can I measure food"         -> "what type of food are you measuring?"
  - "can I measure a wallet"     -> a clarifying question about corner count
  - TVS MSP 250 (a dot matrix)   -> "is an inkjet printer"

Every one of those was previously fixed by adding another regex to _EXCLUDED_CATEGORIES.
That does not converge: the categories are open-ended (every dish, every animal, every
object), so there is always a next miss, and each one reaches the customer as either a
refusal for an innocent question or a confidently wrong answer.

The printer classifier already proved the diagnosis. Same model, same task: asked inside
Pass 2 it scored 10/12 and called a dot matrix an inkjet; asked in isolation with a small
prompt it scored 15/15, self-consistent across three runs. Printers were not special -
ISOLATION was. This generalises that result to the whole rule rather than replicating it
per category.

The existing regexes are not removed. Their role changes: they were the enforcement
mechanism and had to be extended forever; they are now a zero-latency cache in front of
this, for the cases already measured. Anything they do not claim lands here instead of in
Pass 2's improvisation.

LATENCY

This costs nothing on the common path and SAVES time on the path it claims:

  - It only runs when a question is measure-shaped AND neither the out-of-scope nor the
    in-scope cache claimed it - so ordinary wall/room/pricing questions never reach it.
  - It runs concurrently with Pass 1 and retrieval (same pattern as llm/moderation.py),
    so its ~1.3s overlaps work already happening.
  - A "cannot measure" verdict REPLACES Pass 2 instead of preceding it, removing a ~5s
    call. The measured printer equivalent answered in 4.2s against Pass 2's ~7s.

Fails open, like every other optional layer here: any error returns None and the turn
proceeds to Pass 2 exactly as it did before this module existed.
"""
from typing import Optional, Tuple

from llm.client import call_llm

# Condensed from Rule 8/8B/8C - the capability sheet only. Deliberately carries no tone
# instructions, no ticket policy, no safety rules and no retrieved context: the entire
# point is that this decision is made on its own, not alongside five other jobs.
CLASSIFIER_PROMPT = """You decide whether a product called Geometra can measure something.

Geometra measures physical surfaces and objects from phone photos: walls, wall elevations,
ceilings, floors, rooms, halls, doors, windows, wardrobes, cabinets, cupboards, shelves,
countertops, washbasins, staircases, furniture, electrical outlets/sockets, and photo
frames. It needs a flat, closed, non-curved shape.

It CANNOT measure:
  vehicle      - vehicles or anything in motion (cars, buses, trains, bikes, planes, boats)
  living       - anything alive, INCLUDING ALL PLANT LIFE: a person, animal, insect, bird,
                 fish, plant, flower, tree or shrub. A tree is "living", never "natural".
  effigy       - representations of a living thing: mannequins, statues, dolls, stuffed toys
  liquid       - any liquid or a container of one: pools, ponds, lakes, rivers, tanks,
                 bottles, mugs, glasses of liquid (full OR empty)
  natural      - formless natural elements ONLY: rain, wind, fire, dust, mud, sand, snow.
                 Never use this for anything alive - that is "living".
  handheld     - small handheld or loose items: pens, phones, laptops, books, utensils,
                 cosmetics, coins, wallets, bags, luggage, clothes, curtains, toys,
                 printers, microwaves, musical instruments, garbage, and FOOD OR DRINK OF
                 ANY KIND - any dish, meal, snack, sweet or ingredient from any cuisine in
                 the world, whether or not you recognise the name. Indian dishes come up
                 constantly here: biryani, dosa, idli, roti, samosa, paneer, curry, thali
                 are all "handheld", as are pizza, cake, bread and every other food.
  reflective   - reflective or transparent surfaces: mirrors, glass, glass doors/windows
  outdoor      - standalone structures not part of a room: mountains, monuments, towers,
                 poles, streets, buildings seen from outside
  celestial    - celestial bodies: planets, stars, moons, galaxies, black holes
  weapon       - weapons: knives, swords, guns, and hand tools like hammers, screwdrivers
  curved       - a curved or rounded surface, whatever it is made of

Reply with EXACTLY one lowercase word and nothing else:
  - "measurable" if Geometra can measure it
  - one of the category words above if it cannot
  - "unclear" ONLY if the message names no specific thing at all ("can I measure it").
    If it names something you don't recognise, decide from what the word most likely is -
    an unfamiliar dish name is still food, an unfamiliar animal is still living. Do not
    answer "unclear" merely because a word is unfamiliar.

Do not explain. Do not add punctuation. One word only."""

# Maps the classifier's one-word verdict onto the reason strings already used by
# config.EXCLUDED_ITEM_MESSAGE_TEMPLATE, so the customer-facing wording stays in config
# and is identical whether the verdict came from a regex or from here.
CATEGORY_REASONS = {
    "vehicle": "it's a vehicle",
    "living": "it's a living thing",
    "liquid": "it's a liquid or liquid container",
    "natural": "it's a natural element",
    "handheld": "it's a small handheld or loose item",
    "reflective": "it's a reflective or transparent surface",
    "outdoor": "it's a standalone outdoor structure, not part of a room or hall",
    "celestial": "it's a celestial body",
    "weapon": "it's a weapon or tool",
    "curved": "it's a curved surface - Geometra is designed for flat, closed, "
              "non-curved shapes",
}

# "effigy" has its own dedicated message (mannequins/statues/dolls name themselves in it),
# so it is returned as a verdict but not mapped to the generic template.
_VALID = set(CATEGORY_REASONS) | {"effigy", "measurable", "unclear"}


def classify_measurability(message: str) -> Optional[str]:
    """Returns a verdict word - "measurable", "unclear", "effigy", or a CATEGORY_REASONS
    key - or None if the call itself failed.

    None and "unclear" both mean "carry on to Pass 2", but they are different events: the
    model genuinely could not tell, versus the call broke. Both are safe, because Pass 2
    is exactly where the turn went before this module existed.
    """
    try:
        raw, _, _ = call_llm(CLASSIFIER_PROMPT, message)
    except Exception as e:
        print(f"[measurability] classification failed, falling through to Pass 2: {e}")
        return None
    answer = raw.strip().lower().strip(".").split()[0] if raw.strip() else ""
    return answer if answer in _VALID else "unclear"


def reason_for(verdict: str) -> Optional[Tuple[str, bool]]:
    """(reason string, uses_generic_template) for a verdict that excludes, else None."""
    if verdict == "effigy":
        return None, False
    if verdict in CATEGORY_REASONS:
        return CATEGORY_REASONS[verdict], True
    return None
