"""
pipeline/nlp_extractor.py
Uses spaCy NER to extract people, places, organisations and figures.
"""

import spacy


class NLPExtractor:
    """
    Runs spaCy Named Entity Recognition on article text to pull out:

      PERSON            → individuals  (e.g. William Ruto, Angela Oketch)
      GPE / LOC         → locations    (e.g. Nairobi, Turkana, Kenya)
      ORG               → organisations(e.g. KNH, Daily Nation, Cabinet)
      CARDINAL / MONEY
      / PERCENT         → figures      (e.g. Sh400, 50 per cent, 6 people)

    Also extracts keywords via noun-chunk frequency ranking.
    """

    def __init__(self, model: str = "en_core_web_sm"):
        print("[nlp] Loading spaCy model …")
        self.nlp = spacy.load(model)

    def extract(self, text: str) -> dict:
        """
        Accepts a string of article text.
        Returns a dict with individuals, locations,
        organisations, figures, and keywords.
        """
        # Guard against very long texts hitting spaCy's token limit
        doc = self.nlp(text[:100_000])

        individuals    = []
        locations      = []
        organisations  = []
        figures        = []

        for ent in doc.ents:
            label = ent.label_
            value = ent.text.strip()

            if label == "PERSON" and value not in individuals:
                individuals.append(value)

            elif label in ("GPE", "LOC") and value not in locations:
                locations.append(value)

            elif label == "ORG" and value not in organisations:
                organisations.append(value)

            elif label in ("CARDINAL", "MONEY", "PERCENT", "QUANTITY"):
                figures.append({"value": value, "type": label})

        # ── Keyword extraction ─────────────────────────────────────
        # Count how often each noun chunk appears, rank by frequency
        chunk_freq: dict[str, int] = {}
        for chunk in doc.noun_chunks:
            key = chunk.text.lower().strip()
            # Keep only multi-word phrases (more meaningful than single words)
            if 2 <= len(key.split()) <= 4:
                chunk_freq[key] = chunk_freq.get(key, 0) + 1

        # Return top 10 keywords, most frequent first
        keywords = [
            k for k, _ in sorted(
                chunk_freq.items(), key=lambda x: -x[1]
            )[:10]
        ]

        return {
            "individuals":   individuals,
            "locations":     locations,
            "organisations": organisations,
            "figures":       figures,
            "keywords":      keywords,
        }