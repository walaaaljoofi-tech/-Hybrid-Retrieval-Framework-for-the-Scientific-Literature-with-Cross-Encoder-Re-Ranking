# Preparing document texts
# ======================

def build_doc_text(doc):
    """We prefer title + abstract if available; otherwise, we use text."""
    title = doc.get("title", "") or ""
    abstract = doc.get("abstract", "") or ""
    text = doc.get("text", "") or ""
    if title or abstract:
        return f"{title}. {abstract}".strip()
    return text.strip()

doc_ids = list(corpus.keys())
doc_texts = [build_doc_text(corpus[did]) for did in doc_ids]
docid_to_text = dict(zip(doc_ids, doc_texts))

# English stopwords + basic Arabic stopwords
# --------------------------------------------

STOPWORDS_EN = set("""
the a an and or of for to in on at by with from is are were was be been
this that these those as into about using use based
""".split())

STOPWORDS_AR = set("""
من في على إلى التي الذي عن أن إن كان كانت تكون يكون هذا هذه ذلك تلك
هو هي هم هن ثم حيث كما إذا إذ قد لقد لم لن لا ما
""".split())


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # --------------------------------------------
    # 1) Convert English text to lowercase
    # --------------------------------------------
    text = text.lower()

    # --------------------------------------------
    # 2) Remove newline characters
    # --------------------------------------------
    text = text.replace("\n", " ")

    # --------------------------------------------
    # 3) Remove URLs
    # --------------------------------------------
    text = re.sub(r"http\S+|www\S+", " ", text)

    # --------------------------------------------
    # 4) Remove punctuation
    # --------------------------------------------
    text = text.translate(str.maketrans("", "", string.punctuation))

    # --------------------------------------------
    # 5) Remove non-Arabic/non-English characters
    # --------------------------------------------
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)

    # --------------------------------------------
    # 6) Remove extra whitespace
    # --------------------------------------------
    text = re.sub(r"\s+", " ", text).strip()

    return text
   
    def simple_tokenize(text: str):
    return re.findall(r"[a-z0-9]+", str(text).lower())


def build_doc_text_from_beir(doc: dict) -> str:
    """Build document text from a BEIR SciFact document."""
    title = doc.get("title", "")
    text = doc.get("text", "")

    return f"{title} {text}".strip()
