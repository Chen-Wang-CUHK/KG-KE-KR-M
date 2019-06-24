import re
import string

DIGIT = '<digit>'
KEYWORDS_TUNCATE = 10
MAX_KEYWORD_LEN = 6
PRINTABLE = set(string.printable)


def get_tokens(text, fine_grad=True):
    """
    Need use the same word tokenizer between keywords and source context
    keep [_<>,\(\)\.\'%], replace digits to <digit>, split by [^a-zA-Z0-9_<>,\(\)\.\'%]
    """
    text = re.sub(r'[\r\n\t]', '', text)
    text = ''.join(list(filter(lambda x: x in PRINTABLE, text)))
    if fine_grad:
        # tokenize by non-letters
        # Although we have will use corenlp for tokenizing,
        # we still use the following tokenizer for fine granularity
        tokens = filter(lambda w: len(w) > 0, re.split(r'[^a-zA-Z0-9_<>,\(\)\.\'%]', text))
    else:
        tokens = text.split()
    # replace the digit terms with <digit>
    tokens = [w if not re.match('^\d+$', w) else DIGIT for w in tokens]
    return tokens


def process_keyphrase(keyword_str, limit_num=True, fine_grad=True):
    # replace some noise characters
    keyphrases = keyword_str.replace('?', '')
    # retrieved keyphrases are split by '<eos>'
    keyphrases = keyphrases.replace('<eos>', ';')
    # replace abbreviations
    keyphrases = re.sub(r'\(.*?\)', '', keyphrases)
    # Note: keyword should be applied the same tokenizer as the source did
    keyphrases = [get_tokens(keyword.strip(), fine_grad) for keyword in keyphrases.split(';')]

    # ['key1a key1b', 'key2a key2b']
    if limit_num:
        keyphrases = [' '.join(key) for key in keyphrases if 0 < len(key) <= MAX_KEYWORD_LEN]
    else:
        keyphrases = [' '.join(key) for key in keyphrases if 0 < len(key)]

    return keyphrases
