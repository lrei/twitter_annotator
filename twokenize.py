# -*- coding: utf-8 -*-
"""
Twokenize -- a tokenizer designed for Twitter text in English and some other European languages.
This tokenizer code has gone through a long history:

(1) Brendan O'Connor wrote original version in Python, http://github.com/brendano/tweetmotif
       TweetMotif: Exploratory Search and Topic Summarization for Twitter.
       Brendan O'Connor, Michel Krieger, and David Ahn.
       ICWSM-2010 (demo track), http://brenocon.com/oconnor_krieger_ahn.icwsm2010.tweetmotif.pdf
(2a) Kevin Gimpel and Daniel Mills modified it for POS tagging for the CMU ARK Twitter POS Tagger
(2b) Jason Baldridge and David Snyder ported it to Scala
(3) Brendan bugfixed the Scala port and merged with POS-specific changes
    for the CMU ARK Twitter POS Tagger  
(4) Tobi Owoputi ported it back to Java and added many improvements (2012-06)

Current home is http://github.com/brendano/ark-tweet-nlp and http://www.ark.cs.cmu.edu/TweetNLP

There have been at least 2 other Java ports, but they are not in the lineage for the code here.

(5) Ported to Python by Myle Ott <myleott@gmail.com>.

(6) Some hacks by Luis Rei <luis.rei@ijs.si> for adaptation to a specific pipeline
"""

from __future__ import print_function

import sys
import os
import inspect
import struct
import operator
import re
import HTMLParser
import argparse
from functools import partial
from cStringIO import StringIO
from base64 import b64decode
from gzip import GzipFile


def regex_or(*items):
    return '(?:' + '|'.join(items) + ')'


def unichar(i):
    try:
        return unichr(i)
    except ValueError:
        return struct.pack('i', i).decode('utf-32')

#
# Emoji
#
emoji_data = """
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
"""
# we now decode the file's content from the string and unzip it
orig_file_desc = GzipFile(mode='r', fileobj=StringIO(b64decode(emoji_data)))
# get the original's file content to a variable
all_emoji = orig_file_desc.readlines()
# and close the file descriptor
orig_file_desc.close()
all_emoji = [x.strip() for x in all_emoji if not x.strip().startswith('#')]
all_emoji = [x.split(';')[0].strip() for x in all_emoji]
all_emoji_1 = [unichar(int(x, 16)) for x in all_emoji if len(x.split()) == 1]
all_emoji_2 = [x.split() for x in all_emoji if len(x.split()) == 2]
all_emoji_2 = [unichar(int(x[0].strip(), 16)) + unichar(int(x[1].strip(), 16))
               for x in all_emoji_2]
all_emoji = all_emoji_1 + all_emoji_2
all_emoji = [re.escape(x) for x in all_emoji]
all_emoji_str = u'(' + ur'|'.join(all_emoji) + u')'
re_emoji = re.compile(all_emoji_str, re.UNICODE)
#
# End of Emoji
#

Contractions = re.compile(u"(?i)(\w+)(n['’′]t|['’′]ve|['’′]ll|['’′]d|['’′]re|['’′]s|['’′]m)$", re.UNICODE)
Whitespace = re.compile(u"[\s\u0020\u00a0\u1680\u180e\u202f\u205f\u3000\u2000-\u200a]+", re.UNICODE)

punctChars = r"['\"“”‘’.?!…,:;]"
#punctSeq   = punctChars+"+"	#'anthem'. => ' anthem '.
punctSeq   = r"['\"“”‘’]+|[.?!,…]+|[:;]+"	#'anthem'. => ' anthem ' .
entity     = r"&(?:amp|lt|gt|quot);"

# don't it's a trap l'app - words separated by apostrophe
ApWords = re.compile(ur"(\w+)('|\u2019)(\w+)", re.UNICODE)


#  URLs

# BTO 2012-06: everyone thinks the daringfireball regex should be better, but they're wrong.
# If you actually empirically test it the results are bad.
# Please see https://github.com/brendano/ark-tweet-nlp/pull/9

urlStart1  = r"(?:https?://|\bwww\.)"
commonTLDs = r"(?:com|org|edu|gov|net|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|pro|tel|travel|xxx)"
ccTLDs	 = r"(?:ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|" + \
r"bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|" + \
r"er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|" + \
r"hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|" + \
r"lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|" + \
r"nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|sk|" + \
r"sl|sm|sn|so|sr|ss|st|su|sv|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|" + \
r"va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|za|zm|zw)"	#TODO: remove obscure country domains?
urlStart2  = r"\b(?:[A-Za-z\d-])+(?:\.[A-Za-z0-9]+){0,3}\." + regex_or(commonTLDs, ccTLDs) + r"(?:\."+ccTLDs+r")?(?=\W|$)"
urlBody    = r"(?:[^\.\s<>][^\s<>]*?)?"
urlExtraCrapBeforeEnd = regex_or(punctChars, entity) + "+?"
urlEnd     = r"(?:\.\.+|[<>]|\s|$)"
url        = regex_or(urlStart1, urlStart2) + urlBody + "(?=(?:"+urlExtraCrapBeforeEnd+")?"+urlEnd+")"


# Numeric
timeLike   = r"\d+(?::\d+){1,2}"
#numNum     = r"\d+\.\d+"
numberWithCommas = r"(?:(?<!\d)\d{1,3},)+?\d{3}" + r"(?=(?:[^,\d]|$))"
numComb	 = u"[\u0024\u058f\u060b\u09f2\u09f3\u09fb\u0af1\u0bf9\u0e3f\u17db\ua838\ufdfc\ufe69\uff04\uffe0\uffe1\uffe5\uffe6\u00a2-\u00a5\u20a0-\u20b9]?\\d+(?:\\.\\d+)+%?".encode('utf-8')

# Abbreviations
boundaryNotDot = regex_or("$", r"\s", r"[“\"?!,:;]", entity)
aa1  = r"(?:[A-Za-z]\.){2,}(?=" + boundaryNotDot + ")"
aa2  = r"[^A-Za-z](?:[A-Za-z]\.){1,}[A-Za-z](?=" + boundaryNotDot + ")"
standardAbbreviations = r"\b(?:[Mm]r|[Mm]rs|[Mm]s|[Dd]r|[Ss]r|[Jj]r|[Rr]ep|[Ss]en|[Ss]t)\."
arbitraryAbbrev = regex_or(aa1, aa2, standardAbbreviations)
separators  = "(?:--+|―|—|~|–|=)"
decorations = u"(?:[♫♪]+|[★☆]+|[♥❤♡]+|[\u2639-\u263b]+|[\ue001-\uebbb]+)".encode('utf-8')
thingsThatSplitWords = r"[^\s\.,?\"]"
embeddedApostrophe = thingsThatSplitWords+r"+['’′]" + thingsThatSplitWords + "*"

#  Emoticons
# myleott: in Python the (?iu) flags affect the whole expression
#normalEyes = "(?iu)[:=]" # 8 and x are eyes but cause problems
normalEyes = "[:=]" # 8 and x are eyes but cause problems
wink = "[;]"
noseArea = "(?:|-|[^a-zA-Z0-9 ])" # doesn't get :'-(
happyMouths = r"[D\)\]\}]+"
sadMouths = r"[\(\[\{]+"
tongue = "[pPd3]+"
otherMouths = r"(?:[oO]+|[/\\]+|[vV]+|[Ss]+|[|]+)" # remove forward slash if http://'s aren't cleaned

# mouth repetition examples:
# @aliciakeys Put it in a love song :-))
# @hellocalyclops =))=))=)) Oh well

# myleott: try to be as case insensitive as possible, but still not perfect, e.g., o.O fails
#bfLeft = u"(♥|0|o|°|v|\\$|t|x|;|\u0ca0|@|ʘ|•|・|◕|\\^|¬|\\*)".encode('utf-8')
bfLeft = u"(♥|0|[oO]|°|[vV]|\\$|[tT]|[xX]|;|\u0ca0|@|ʘ|•|・|◕|\\^|¬|\\*)".encode('utf-8')
bfCenter = r"(?:[\.]|[_-]+)"
bfRight = r"\2"
s3 = r"(?:--['\"])"
s4 = r"(?:<|&lt;|>|&gt;)[\._-]+(?:<|&lt;|>|&gt;)"
s5 = "(?:[.][_]+[.])"
# myleott: in Python the (?i) flag affects the whole expression
#basicface = "(?:(?i)" +bfLeft+bfCenter+bfRight+ ")|" +s3+ "|" +s4+ "|" + s5
basicface = "(?:" +bfLeft+bfCenter+bfRight+ ")|" +s3+ "|" +s4+ "|" + s5

eeLeft = r"[＼\\ƪԄ\(（<>;ヽ\-=~\*]+"
eeRight= u"[\\-=\\);'\u0022<>ʃ）/／ノﾉ丿╯σっµ~\\*]+".encode('utf-8')
eeSymbol = r"[^A-Za-z0-9\s\(\)\*:=-]"
eastEmote = eeLeft + "(?:"+basicface+"|" +eeSymbol+")+" + eeRight

oOEmote = r"(?:[oO]" + bfCenter + r"[oO])"


emoticon = regex_or(
        # Standard version  :) :( :] :D :P
        "(?:>|&gt;)?" + regex_or(normalEyes, wink) + regex_or(noseArea,"[Oo]") + regex_or(tongue+r"(?=\W|$|RT|rt|Rt)", otherMouths+r"(?=\W|$|RT|rt|Rt)", sadMouths, happyMouths),

        # reversed version (: D:  use positive lookbehind to remove "(word):"
        # because eyes on the right side is more ambiguous with the standard usage of : ;
        regex_or("(?<=(?: ))", "(?<=(?:^))") + regex_or(sadMouths,happyMouths,otherMouths) + noseArea + regex_or(normalEyes, wink) + "(?:<|&lt;)?",

        #inspired by http://en.wikipedia.org/wiki/User:Scapler/emoticons#East_Asian_style
        eastEmote.replace("2", "1", 1), basicface,
        # iOS 'emoji' characters (some smileys, some symbols) [\ue001-\uebbb]  
        # TODO should try a big precompiled lexicon from Wikipedia, Dan Ramage told me (BTO) he does this

        # myleott: o.O and O.o are two of the biggest sources of differences
        #          between this and the Java version. One little hack won't hurt...
        oOEmote
)

Hearts = "(?:<+/?3+)+" #the other hearts are in decorations

Arrows = regex_or(r"(?:<*[-―—=]*>+|<+[-―—=]*>*)", u"[\u2190-\u21ff]+".encode('utf-8'))

# BTO 2011-06: restored Hashtag, AtMention protection (dropped in original scala port) because it fixes
# "hello (#hashtag)" ==> "hello (#hashtag )"  WRONG
# "hello (#hashtag)" ==> "hello ( #hashtag )"  RIGHT
# "hello (@person)" ==> "hello (@person )"  WRONG
# "hello (@person)" ==> "hello ( @person )"  RIGHT
# ... Some sort of weird interaction with edgepunct I guess, because edgepunct 
# has poor content-symbol detection.

# This also gets #1 #40 which probably aren't hashtags .. but good as tokens.
# If you want good hashtag identification, use a different regex.
Hashtag = "#[a-zA-Z0-9_]+"  #optional: lookbehind for \b
#optional: lookbehind for \b, max length 15
AtMention = "[@＠][a-zA-Z0-9_]+"

# I was worried this would conflict with at-mentions
# but seems ok in sample of 5800: 7 changes all email fixes
# http://www.regular-expressions.info/email.html
Bound = r"(?:\W|^|$)"
Email = regex_or("(?<=(?:\W))", "(?<=(?:^))") + r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}(?=" +Bound+")"

# We will be tokenizing using these regexps as delimiters
# Additionally, these things are "protected", meaning they shouldn't be further split themselves.
Protected  = re.compile(
    unicode(regex_or(
        Hearts,
        url,
        Email,
        timeLike,
        #numNum,
        numberWithCommas,
        numComb,
        emoticon,
        Arrows,
        entity,
        punctSeq,
        arbitraryAbbrev,
        separators,
        decorations,
        embeddedApostrophe,
        Hashtag,  
        AtMention
    ).decode('utf-8')), re.UNICODE)

url_re = re.compile(url, re.UNICODE)
mention_re = re.compile(AtMention, re.UNICODE)

# Edge punctuation
# Want: 'foo' => ' foo '
# While also:   don't => don't
# the first is considered "edge punctuation".
# the second is word-internal punctuation -- don't want to mess with it.
# BTO (2011-06): the edgepunct system seems to be the #1 source of problems these days.  
# I remember it causing lots of trouble in the past as well.  Would be good to revisit or eliminate.

# Note the 'smart quotes' (http://en.wikipedia.org/wiki/Smart_quotes)
#edgePunctChars    = r"'\"“”‘’«»{}\(\)\[\]\*&" #add \\p{So}? (symbols)
edgePunctChars    = u"'\"“”‘’«»{}\\(\\)\\[\\]\\*&" #add \\p{So}? (symbols)
edgePunct    = "[" + edgePunctChars + "]"
notEdgePunct = "[a-zA-Z0-9]" # content characters
offEdge = r"(^|$|:|;|\s|\.|,)"  # colon here gets "(hello):" ==> "( hello ):"
EdgePunctLeft  = re.compile(offEdge + "("+edgePunct+"+)("+notEdgePunct+")", re.UNICODE)
EdgePunctRight = re.compile("("+notEdgePunct+")("+edgePunct+"+)" + offEdge, re.UNICODE)

# number
# numbers can include .,/ e.g. 12,399.05 or 12.2.2005 or 12/2/2005
base_number = r'\d+((\,\d+)*(\.\d+)*(\/\d+)*)*'
number = r'\b' + base_number + r'\b'
num_re = re.compile(number, re.UNICODE)

# hastag
hash_re = re.compile(r'(\A|\s)#(\w+)', re.UNICODE)


def splitEdgePunct(input):
    input = EdgePunctLeft.sub(r"\1\2 \3", input)
    input = EdgePunctRight.sub(r"\1 \2\3", input)
    return input

# The main work of tokenizing a tweet.
def simpleTokenize(text):

    # Do the no-brainers first
    splitPunctText = splitEdgePunct(text)

    textLength = len(splitPunctText)
    
    # BTO: the logic here got quite convoluted via the Scala porting detour
    # It would be good to switch back to a nice simple procedural style like in the Python version
    # ... Scala is such a pain.  Never again.

    # Find the matches for subsequences that should be protected,
    # e.g. URLs, 1.0, U.N.K.L.E., 12:53
    bads = []
    badSpans = []
    for match in Protected.finditer(splitPunctText):
        # The spans of the "bads" should not be split.
        if (match.start() != match.end()): #unnecessary?
            bads.append( [splitPunctText[match.start():match.end()]] )
            badSpans.append( (match.start(), match.end()) )

    # Create a list of indices to create the "goods", which can be
    # split. We are taking "bad" spans like 
    #     List((2,5), (8,10)) 
    # to create 
    #     List(0, 2, 5, 8, 10, 12)
    # where, e.g., "12" here would be the textLength
    # has an even length and no indices are the same
    indices = [0]
    for (first, second) in badSpans:
        indices.append(first)
        indices.append(second)
    indices.append(textLength)

    # Group the indices and map them to their respective portion of the string
    splitGoods = []
    for i in range(0, len(indices), 2):
        goodstr = splitPunctText[indices[i]:indices[i+1]]
        splitstr = goodstr.strip().split(" ")
        splitGoods.append(splitstr)

    #  Reinterpolate the 'good' and 'bad' Lists, ensuring that
    #  additonal tokens from last good item get included
    zippedStr = []
    for i in range(len(bads)):
        zippedStr = addAllnonempty(zippedStr, splitGoods[i])
        zippedStr = addAllnonempty(zippedStr, bads[i])
    zippedStr = addAllnonempty(zippedStr, splitGoods[len(bads)])

    # BTO: our POS tagger wants "ur" and "you're" to both be one token.
    # Uncomment to get "you 're"
    #splitStr = []
    #for tok in zippedStr:
    #    splitStr.extend(splitToken(tok))
    #zippedStr = splitStr

    # fix emoji tokenization
    splitStr = []
    for tok in zippedStr:
        splitStr.extend(re_emoji.split(tok))
    zippedStr = splitStr
    zippedStr = u' '.join(splitStr).split()
    
    return zippedStr

def addAllnonempty(master, smaller):
    for s in smaller:
        strim = s.strip()
        if (len(strim) > 0):
            master.append(strim)
    return master

# "foo   bar " => "foo bar"
def squeezeWhitespace(input):
    return Whitespace.sub(" ", input).strip()

# Final pass tokenization based on special patterns
def splitToken(token):
    m = Contractions.search(token)
    if m:
        return [m.group(1), m.group(2)]
    return [token]

# Assume 'text' has no HTML escaping.
def tokenize1(text):
    return simpleTokenize(squeezeWhitespace(text))


def tokenize2(text):
    """Breaks apostrophes:
        l'ammore -> ["l'", "ammore"]
        """
    tokens = simpleTokenize(squeezeWhitespace(text))
    ntoks = []
    for tok in tokens:
        if '\'' in tok:
            matching = ApWords.match(tok)
            if matching is not None:
                tok_list = list(matching.groups())
                p1 = tok_list[0] + tok_list[1]  # "l'"
                ntoks.extend([p1])
                ntoks.extend(tok_list[2:])  # "ammore"
        else:
            ntoks.extend([tok])
    return ntoks

def tokenize(text, break_apostrophes=False):
    """Returns tokenized text
    Expects unicode or utf8 encoded text
    Returns unicode string (the tokenized text seperated by space)
    """
    tokens = []
    text = text.strip()

    if not isinstance(text, unicode):
        text = text.decode('utf8')
    if not text:
        return ''

    text = text.replace(u'&amp;', u'&')

    # tokenize
    if break_apostrophes:
        tokens = tokenize2(text)
    else:
        tokens = tokenize1(text)

    text = u' '.join(tokens)

    return text


# Partial
tokenize_apostrophes = partial(tokenize, break_apostrophes=True)


# Twitter text comes HTML-escaped, so unescape it.
# We also first unescape &amp;'s, in case the text has been buggily double-escaped.
def normalizeTextForTagger(text):
    text = text.replace("&amp;", "&")
    text = HTMLParser.HTMLParser().unescape(text)
    return text

# This is intended for raw tweet text -- we do some HTML entity unescaping before running the tagger.
# 
# This function normalizes the input text BEFORE calling the tokenizer.
# So the tokens you get back may not exactly correspond to
# substrings of the original text.
def tokenizeRawTweetText(text):
    tokens = tokenize(normalizeTextForTagger(text))
    return tokens


def preprocess(tokenized_text):
    """Returns unicode string
    """
    text = tokenized_text
    if not text:
        return text

    if not isinstance(text, unicode):
        text = text.decode('utf8')

    text = num_re.sub('tnumnum', text)
    text = url_re.sub('turlurl', text)
    text = mention_re.sub('tuseruser', text)
    text = hash_re.sub(r' # \2', text)
    text = text.lower()
    text = text.strip()

    return text  


def main():
    parser = argparse.ArgumentParser(description='Run tokenizer (twokenize).')
    parser.add_argument('input_file', help='the input file')
    parser.add_argument('output_file', help='the output file')

    parser.add_argument('--tsv', action='store_true', default=False,
                        help='input and output are TSV files not text')
    parser.add_argument('--preprocess', action='store_true', default=True,
                        help='preprocess text')
    parser.add_argument('--apostrophes', action='store_true',
                        default=False, help="don't becomes d' ont")
    parser.add_argument('--ignore', action='store_true',
                        default=False, help='ignores if in/out text is empty')

    # Parse
    args = parser.parse_args()
    infile = args.input_file
    outfile = args.output_file
    ignore = args.ignore


    with open(infile) as fin, open(outfile, 'w') as fout:
        for line in fin:
            if args.tsv:
                line = line.decode('utf8')
                fields = line.split(u'\t')
                line = fields[0]
                others = u'\t'.join(fields[1:]).strip().encode('utf8')

            text = tokenize(line, args.apostrophes)
            if args.preprocess:
                text = preprocess(text)

            if not text and not ignore:
                print('Empty line in result')
                print(line)
                sys.exit(1)

            elif not text and ignore:
                continue

            text = text.encode('utf8')
            if args.tsv:
                fout.write(text + '\t' + others + '\n')
            else:
                fout.write(text + '\n')

if __name__ == '__main__':
    main()
