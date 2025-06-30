import re
import nltk
import string

STOPWORDS = list(nltk.corpus.stopwords.words("english"))
PUNCTUATION = string.punctuation + "’…"
REPL_PUNC = re.compile(rf"[{PUNCTUATION}]+")


def adapt_tags(tag):
    # The PerceptronTagger outputs tags in a different format than
    # the WordNetLemmatizer was trained-on, so adjust accordingly
    if tag.startswith("J"):
        return nltk.corpus.wordnet.ADJ
    if tag.startswith("V"):
        return nltk.corpus.wordnet.VERB
    if tag.startswith("N"):
        return nltk.corpus.wordnet.NOUN
    if tag.startswith("R"):
        return nltk.corpus.wordnet.ADV
    return nltk.corpus.wordnet.NOUN


def remove_punctuation(tokens):
    out = []
    for token in tokens:
        token = REPL_PUNC.sub("", token)
        if len(token) > 0:
            out.append(token)
    return out


def tokenize_and_clean(text):
    # Tokenize the lower-case text
    tokens = nltk.word_tokenize(text)
    # Extract the tag for each token
    # Note: pos_tag is much faster than PerceptronTagger
    pairs = nltk.pos_tag(tokens)
    # Lemmatize tokens using tags as hints
    lems = [
        nltk.stem.WordNetLemmatizer().lemmatize(pair[0], adapt_tags(pair[1]))
        for pair in pairs
    ]
    # Remove stopwords
    unstopped = [word for word in lems if word not in STOPWORDS]
    # Remove puncutation
    nopunc = remove_punctuation(unstopped)
    # Remove short words/single characters
    noshort = [word for word in nopunc if len(word) > 1]

    return noshort


def to_tokens(df, remove_short=True):
    tmp = df.str.lower().str.split(r"[^a-z0-9]+")
    if remove_short:
        return [
            [val.strip() for val in t if len(val.strip()) > 1] for t in tmp.tolist()
        ]
    else:
        return tmp


def str_to_unique_tokens(df):
    unique_words = to_tokens(df, remove_short=False).explode().value_counts()
    unique_words = unique_words[~unique_words.index.isin(STOPWORDS)]
    unique_words = unique_words[unique_words.index.str.len() > 1]
    return unique_words
