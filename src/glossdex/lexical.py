"""Keyword evidence over item texts: BM25, rarity-weighted coverage, and the all-terms test.

The lexical side of glossdex never orders results on its own. It answers three narrower
questions for the result-set stages in :mod:`glossdex.scoring`:

* does the collection contain the query's *subject* at all (the evidence gate),
* how much of the query does each item account for (coverage, used as a capped boost and to
  measure whether one item answers the query far better than the rest),
* may a near-miss item be used to pad a short list.

One lexical document per *item*, built from the item's distinct texts, so an item with more
phrases does not accumulate more hits for the same content.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set

K1 = 1.2
B = 0.75

#: Words shorter than this never count as content. 2 keeps "tv"; the two-letter function words
#: are listed in STOPWORDS explicitly instead.
MIN_CONTENT_LENGTH = 2

_SPLIT = re.compile(r"[\W_]+", re.UNICODE)


def tokenize(text: str) -> List[str]:
    """Lowercase, split on anything that is not a letter or digit, drop single characters.

    Numbers are kept on purpose: invoice numbers, years and amounts are among the most
    discriminative terms in a collection of documents.
    """
    return [t for t in _SPLIT.split(text.lower()) if len(t) > 1]


#: Words that say nothing about WHICH item is meant. Words that describe something visible
#: ("red", "left", "old", "small") are deliberately absent.
STOPWORDS: Set[str] = {
    # articles, conjunctions, particles
    "a", "an", "the", "and", "or", "nor", "but", "if", "then", "than", "so", "because",
    "as", "while", "although", "though", "whether", "either", "neither",
    # prepositions and positional words
    "of", "in", "on", "at", "to", "for", "with", "by", "from", "into", "onto", "upon",
    "over", "under", "above", "below", "beneath", "beside", "between", "among", "around",
    "through", "across", "along", "behind", "front", "top", "bottom", "side", "near",
    "next", "up", "down", "off", "out", "inside", "outside", "within", "without",
    "toward", "towards", "against", "about", "during", "before", "after", "until",
    # pronouns and determiners
    "my", "mine", "me", "we", "us", "our", "ours", "you", "your", "yours",
    "he", "him", "his", "she", "her", "hers", "it", "its", "they", "them",
    "their", "theirs", "who", "whom", "whose", "which", "what", "that", "this",
    "these", "those", "there", "here", "where", "when", "why", "how",
    "some", "any", "all", "both", "each", "every", "few", "many", "much", "more",
    "most", "several", "other", "another", "same", "such", "own", "no", "none",
    # verbs that carry no content
    "is", "are", "was", "were", "be", "been", "being", "am", "do", "does", "did",
    "have", "has", "had", "having", "can", "could", "will", "would", "shall",
    "should", "may", "might", "must", "get", "got", "make", "makes", "made",
    "take", "takes", "show", "shows", "showing", "showed", "see", "seeing", "seen",
    "look", "looking", "looks", "contain", "contains", "containing",
    # adverbs and fillers
    "not", "very", "just", "only", "also", "too", "quite", "rather", "really",
    "again", "still", "even", "ever", "never", "always", "usually", "maybe",
    "perhaps", "somewhere", "anywhere", "everywhere", "somehow",
    # search-request phrasing: typed AT a search box, not about an item
    "photo", "photos", "picture", "pictures", "image", "images", "pic", "pics",
    "find", "search", "want", "need", "please", "give", "list",
}


def content_terms(query: str) -> List[str]:
    """The query's meaning-bearing words: tokenized, de-duplicated, stopwords removed."""
    seen: Dict[str, None] = {}
    for t in tokenize(query):
        if len(t) >= MIN_CONTENT_LENGTH and t not in STOPWORDS:
            seen.setdefault(t, None)
    return list(seen)


#: Equivalence classes bridging the words people TYPE to the words a describing model WRITES,
#: about the same thing. Membership is per class and NOT transitive: "wife" and "mother" both
#: reach "woman" but never each other. A loose member manufactures false full matches, so the
#: rule is: when in doubt, leave a word out.
SYNONYM_CLASSES: List[Set[str]] = [
    # people: relationship words bridge to the observable word, never to each other
    {"wife", "wives", "woman", "women", "lady", "ladies", "female", "females"},
    {"mother", "mothers", "mom", "moms", "mum", "mums",
     "woman", "women", "lady", "ladies", "female", "females"},
    {"husband", "husbands", "man", "men", "guy", "guys", "male", "males"},
    {"father", "fathers", "dad", "dads", "man", "men", "guy", "guys", "male", "males"},
    {"son", "sons", "boy", "boys", "child", "children", "kid", "kids",
     "baby", "babies", "infant", "infants", "toddler", "toddlers"},
    {"daughter", "daughters", "girl", "girls", "child", "children", "kid", "kids",
     "baby", "babies", "infant", "infants", "toddler", "toddlers"},
    {"baby", "babies", "infant", "infants", "newborn", "newborns",
     "toddler", "toddlers", "child", "children", "kid", "kids"},
    {"parents", "adults", "grownups", "grownup", "adult"},
    {"grandmother", "grandmothers", "grandma", "granny", "nana"},
    {"grandfather", "grandfathers", "grandpa", "granddad", "grandad"},
    {"colleague", "colleagues", "coworker", "coworkers", "workmate", "workmates"},
    {"friend", "friends", "buddy", "buddies"},
    {"family", "families", "relatives"},
    {"person", "persons", "people", "someone", "individual", "individuals", "human", "humans"},
    {"doctor", "doctors", "physician", "physicians", "md"},
    # pets
    {"dog", "dogs", "puppy", "puppies", "canine"},
    {"cat", "cats", "kitten", "kittens", "feline"},
    # rooms and places
    {"home", "house", "houses", "residence", "apartment", "flat"},
    {"bedroom", "bedrooms"},
    {"bathroom", "bathrooms", "washroom", "restroom", "toilet"},
    {"kitchen", "kitchens"},
    {"living", "lounge", "livingroom"},
    {"office", "offices", "workspace", "workstation"},
    {"garden", "gardens", "yard", "backyard", "lawn"},
    {"mall", "malls", "shopping"},
    {"store", "stores", "shop", "shops", "market", "supermarket", "grocery"},
    {"restaurant", "restaurants", "cafe", "diner", "eatery", "bistro"},
    {"park", "parks", "playground", "playgrounds"},
    {"beach", "beaches", "seaside", "shore", "coast"},
    {"pool", "pools", "swimming"},
    {"school", "schools", "classroom"},
    {"hospital", "hospitals", "clinic"},
    {"airport", "airports", "tarmac", "runway"},
    {"hotel", "hotels", "resort", "motel"},
    {"zoo", "zoos", "safari"},
    {"street", "streets", "road", "roads", "sidewalk", "pavement"},
    # objects
    {"couch", "couches", "sofa", "sofas", "settee"},
    {"television", "televisions", "tv", "tvs", "telly"},
    {"computer", "computers", "laptop", "laptops", "macbook"},
    {"monitor", "monitors", "screen", "screens"},
    {"phone", "phones", "mobile", "smartphone", "cellphone", "iphone"},
    {"car", "cars", "vehicle", "vehicles", "automobile", "auto", "sedan", "suv"},
    {"bike", "bikes", "bicycle", "bicycles", "cycling"},
    {"motorcycle", "motorcycles", "motorbike"},
    {"stroller", "strollers", "pram", "pushchair", "buggy"},
    {"bag", "bags", "purse", "purses", "handbag", "backpack", "rucksack"},
    {"glasses", "spectacles", "eyeglasses"},
    {"chair", "chairs", "armchair", "stool"},
    {"table", "tables", "desk", "desks"},
    {"sign", "signs", "signage", "banner", "banners", "placard"},
    {"toy", "toys", "plaything", "playthings"},
    {"bottle", "bottles", "flask"},
    {"book", "books"},
    {"window", "windows"},
    {"door", "doors", "doorway"},
    {"box", "boxes", "carton", "cartons"},
    # documents
    {"document", "documents", "paper", "papers", "form", "forms", "certificate", "certificates"},
    {"licence", "license", "licences", "licenses", "permit"},
    {"passport", "passports"},
    {"bill", "bills", "invoice", "invoices", "statement", "statements", "receipt", "receipts"},
    {"prescription", "prescriptions", "medication", "medications",
     "medicine", "medicines", "pharmacy", "pharmacies"},
    {"identification", "id", "badge", "badges"},
    {"contract", "contracts", "agreement", "agreements"},
    # events
    {"party", "parties", "celebration", "celebrations"},
    {"birthday", "birthdays", "bday"},
    {"wedding", "weddings", "marriage", "bridal"},
    {"vacation", "vacations", "holiday", "holidays", "trip", "trips"},
    {"christmas", "xmas", "yuletide"},
    {"concert", "concerts", "gig", "gigs", "performance", "performances"},
    {"festival", "festivals"},
    {"graduation", "graduations"},
    # food
    {"cake", "cakes", "gateau"},
    {"cupcake", "cupcakes", "muffin", "muffins"},
    {"food", "foods", "meal", "meals", "dinner", "lunch", "breakfast"},
    {"drink", "drinks", "beverage", "beverages"},
    {"wine", "wines", "vino"},
    # outdoors and weather
    {"waterfall", "waterfalls", "falls", "cascade"},
    {"mountain", "mountains", "peak", "peaks", "hill", "hills"},
    {"snow", "snowy", "snowing"},
    {"rain", "rainy", "raining"},
    {"sun", "sunny", "sunshine", "sunlight"},
    {"lake", "lakes", "river", "rivers", "sea", "seas", "ocean", "oceans"},
    {"tree", "trees"},
    {"flower", "flowers", "floral", "bloom", "blooms"},
    {"boat", "boats", "ship", "ships", "sailboat", "yacht"},
    {"plane", "planes", "airplane", "aeroplane", "aircraft", "jet"},
]


def _irregular() -> Dict[str, str]:
    out: Dict[str, str] = {}

    def v(base: str, *forms: str) -> None:
        for f in forms:
            out[f] = base

    # Deliberately NOT stand/standing: a TV stand and a person standing are different things.
    v("hold", "held", "holding", "holds")
    v("sit", "sat", "sitting", "sits")
    v("eat", "ate", "eaten", "eating", "eats")
    v("drink", "drank", "drunk", "drinking", "drinks")
    v("wear", "wore", "worn", "wearing", "wears")
    v("run", "ran", "running", "runs")
    v("swim", "swam", "swum", "swimming", "swims")
    v("ride", "rode", "ridden", "riding", "rides")
    v("drive", "drove", "driven", "driving", "drives")
    v("take", "took", "taken", "taking", "takes")
    v("give", "gave", "given", "giving", "gives")
    v("make", "made", "making", "makes")
    v("blow", "blew", "blown", "blowing", "blows")
    v("throw", "threw", "thrown", "throwing", "throws")
    v("catch", "caught", "catching", "catches")
    v("buy", "bought", "buying", "buys")
    v("cut", "cutting", "cuts")
    v("build", "built", "building", "builds")
    v("sleep", "slept", "sleeping", "sleeps")
    v("feed", "fed", "feeding", "feeds")
    v("lie", "lay", "lying", "lies")
    v("wave", "waved", "waving", "waves")
    v("smile", "smiled", "smiling", "smiles")
    v("laugh", "laughed", "laughing", "laughs")
    v("play", "played", "playing", "plays")
    v("dance", "danced", "dancing", "dances")
    v("cook", "cooked", "cooking", "cooks")
    v("pose", "posed", "posing", "poses")
    v("carry", "carried", "carrying", "carries")
    v("pay", "paid", "paying", "pays")
    v("child", "children")
    v("person", "people")
    v("foot", "feet")
    v("tooth", "teeth")
    v("man", "men")
    v("woman", "women")
    return out


IRREGULAR: Dict[str, str] = _irregular()

_IRREGULAR_FORMS: Dict[str, Set[str]] = defaultdict(set)
for _form, _base in IRREGULAR.items():
    _IRREGULAR_FORMS[_base].add(_form)
for _base in list(_IRREGULAR_FORMS):
    _IRREGULAR_FORMS[_base].add(_base)


def stem(word: str) -> str:
    """A deliberately small normalizer: irregular forms by table, then plural suffixes only."""
    base = IRREGULAR.get(word)
    if base is not None:
        return base
    if len(word) <= 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ss") or word.endswith("us"):
        return word
    if word.endswith("s"):
        return word[:-1]
    return word


class Vocabulary:
    """Synonym classes compiled for lookup. Extend with :meth:`with_classes`."""

    def __init__(self, classes: Sequence[Set[str]]):
        self.classes = [set(c) for c in classes]
        by_word: Dict[str, Set[str]] = {}
        for cls in self.classes:
            for w in cls:
                by_word.setdefault(w, {w}).update(cls)
        self._expansions = by_word

    def with_classes(self, extra: Iterable[Iterable[str]]) -> "Vocabulary":
        return Vocabulary(self.classes + [set(w.lower() for w in c) for c in extra])

    def interchangeable(self, term: str) -> Set[str]:
        """Every word that could stand in for ``term``: synonyms plus number variants.

        Only plural suffixes are generated blindly; verb forms come only from the curated
        irregular table, because "-ing" can turn a noun into an unrelated verb sense.
        """
        out = {term}
        base = stem(term)
        out.add(base)
        out.update(_IRREGULAR_FORMS.get(base, ()))
        if len(base) > 2:
            out.add(base + "s")
            out.add(base + "es")
        out.update(self._expansions.get(term, ()))
        out.update(self._expansions.get(base, ()))
        return out


DEFAULT_VOCABULARY = Vocabulary(SYNONYM_CLASSES)


class LexicalIndex:
    """BM25 postings plus the coverage measures, one document per item."""

    def __init__(
        self,
        postings: Mapping[str, Mapping[int, int]],
        doc_len: Mapping[int, int],
        vocabulary: Vocabulary = DEFAULT_VOCABULARY,
    ):
        self.postings = postings
        self.doc_len = doc_len
        self.doc_count = len(doc_len)
        self.avg_doc_len = (sum(doc_len.values()) / len(doc_len)) if doc_len else 0.0
        self.vocabulary = vocabulary

    @classmethod
    def build(
        cls,
        texts_by_item: Mapping[int, Iterable[str]],
        vocabulary: Vocabulary = DEFAULT_VOCABULARY,
    ) -> "LexicalIndex":
        postings: Dict[str, Dict[int, int]] = defaultdict(dict)
        doc_len: Dict[int, int] = {}
        for item_id, texts in texts_by_item.items():
            # Distinct strings only: the whole description repeats the phrases, and counting it
            # twice would make term frequency meaningless.
            distinct = list(dict.fromkeys(texts))
            terms = [t for s in distinct for t in tokenize(s)]
            doc_len[item_id] = len(terms)
            for t in terms:
                p = postings[t]
                p[item_id] = p.get(item_id, 0) + 1
        return cls(dict(postings), doc_len, vocabulary)

    # -- helpers ---------------------------------------------------------------------------

    def _idf(self, df: int) -> float:
        # Lucene's smoothed IDF. The max() can never fire; kept for parity with the reference.
        return max(0.0, math.log(1.0 + (self.doc_count - df + 0.5) / (df + 0.5)))

    def _matching(self, term: str) -> Set[int]:
        out: Set[int] = set()
        for w in self.vocabulary.interchangeable(term):
            p = self.postings.get(w)
            if p:
                out.update(p.keys())
        return out

    # -- the measures ----------------------------------------------------------------------

    def bm25(self, query: str) -> Dict[int, float]:
        """BM25 per item, normalized by the best score the query COULD achieve.

        The ceiling charges every query term, including terms absent from the collection (at
        the IDF of a term occurring once), so a score near 1 requires matching the query's rare
        terms rather than leading a field of weak matches.
        """
        if self.doc_count == 0:
            return {}
        terms = list(dict.fromkeys(tokenize(query)))
        if not terms:
            return {}
        max_idf = self._idf(1)
        raw: Dict[int, float] = defaultdict(float)
        ceiling = 0.0
        for term in terms:
            posting = self.postings.get(term)
            if posting is None:
                ceiling += max_idf
                continue
            idf = self._idf(len(posting))
            if idf <= 0.0:
                continue
            ceiling += idf
            for item_id, tf in posting.items():
                length = self.doc_len.get(item_id)
                if length is None:
                    continue
                norm = tf * (K1 + 1) / (tf + K1 * (1 - B + B * length / self.avg_doc_len))
                raw[item_id] += idf * norm
        if ceiling <= 0.0 or not raw:
            return {}
        return {k: min(1.0, max(0.0, v / ceiling)) for k, v in raw.items()}

    def coverage(self, query: str) -> Dict[int, float]:
        """Unweighted coverage: the fraction of content terms an item contains (synonyms allowed)."""
        terms = content_terms(query)
        if not terms:
            return {}
        hits: Dict[int, int] = defaultdict(int)
        for t in terms:
            for item_id in self._matching(t):
                hits[item_id] += 1
        n = float(len(terms))
        return {k: v / n for k, v in hits.items()}

    def idf_coverage(self, query: str) -> Dict[int, float]:
        """Rarity-weighted coverage in [0, 1].

        ``C(item) = sum(idf(df_t) for matched t) / (|T| * idf(1))`` where ``df_t`` counts items
        matching any interchangeable form of ``t``. The absolute ceiling keeps a full match on
        rare words near 1 and a full match on common words low.
        """
        terms = content_terms(query)
        if not terms or self.doc_count == 0:
            return {}
        ceiling = len(terms) * self._idf(1)
        if ceiling <= 0.0:
            return {}
        acc: Dict[int, float] = defaultdict(float)
        for t in terms:
            matching = self._matching(t)
            if not matching:
                continue
            w = self._idf(len(matching))
            if w <= 0.0:
                continue
            for item_id in matching:
                acc[item_id] += w
        return {k: min(1.0, max(0.0, v / ceiling)) for k, v in acc.items()}

    def items_matching_at_least(self, query: str, n: int) -> Set[int]:
        """Items containing at least ``n`` content terms, counted once per query term."""
        terms = content_terms(query)
        if not terms or n <= 0:
            return set()
        hits: Dict[int, int] = defaultdict(int)
        for t in terms:
            for item_id in self._matching(t):
                hits[item_id] += 1
        return {k for k, v in hits.items() if v >= n}

    def items_matching_all_terms(self, query: str) -> Set[int]:
        """Items containing EVERY content term, each directly or through a synonym class."""
        terms = content_terms(query)
        if not terms:
            return set()
        candidates: Optional[Set[int]] = None
        for t in terms:
            with_term = self._matching(t)
            if not with_term:
                return set()
            candidates = with_term if candidates is None else candidates & with_term
            if not candidates:
                return set()
        return candidates or set()
