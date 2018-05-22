# encoding: utf-8
from chardet import detect

def encode_utf8(sen):
    if not isinstance(sen, unicode):
        enc = detect(sen)['encoding']
        if enc != 'utf-8':
            sen = sen.decode(enc).encode('utf-8')
    else:
        sen.encode('utf-8')
    return sen

def decode_any(word):
    if not isinstance(word, unicode):
        enc = detect(word)['encoding']
        word = word.decode(enc)
    return word
