# encoding: utf-8
import sys
from chardet import detect
default_encoding = 'utf-8'
if sys.getdefaultencoding() != default_encoding:
    reload(sys)
    sys.setdefaultencoding(default_encoding)

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
