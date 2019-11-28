import spacy
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk

from nltk import word_tokenize
from nltk import pos_tag
from pathlib import Path
from CorpusReader import CorpusReader
from ExtractFeatures import ExtractFeatures
from spacy.lang.en import English



class NlpPipeline:

    wordnet_tag_map = {
        'NN': 'n',
        'NNS': 'n',
        'NNP': 'n',
        'NNPS': 'n',
        'JJ': 'a',
        'JJR': 'a',
        'JJS': 'a',
        'RB': 'r',
        'RBR': 'r',
        'RBS': 'r',
        'VB': 'v',
        'VBD': 'v',
        'VBG': 'v',
        'VBN': 'v',
        'VBP': 'v',
        'VBZ': 'v'
    }

    def createTokens(self,nlp,sentence):
        doc = nlp(sentence)
        for token in doc:
            print(token.text)

    def createLemma(self,nlp,sentence):
        doc = nlp(sentence)
        for token in doc:
            print(token.text,token.lemma_)

    def createPOS(self,nlp,sentence):
        doc = nlp(sentence)
        for token in doc:
            print(token.text, token.pos_)

    def createDepParse(self,nlp,sentence):
        doc = nlp(sentence)
        for token in doc:
            print(token.text, token.dep_, token.head.text, token.head.pos_,
                  [child for child in token.children])

    def createAllNyms(self,sentence):
        wordTokens = word_tokenize(sentence)
        pos = pos_tag(wordTokens)

        hyper = {}
        hypo = {}
        mero = {}
        holo = {}

        index = 0
        for tokenOrig in wordTokens:
            #print(tokenOrig)
            if(pos[index][1] in self.wordnet_tag_map):
                token = lesk(wordTokens, tokenOrig, self.wordnet_tag_map[pos[index][1]])
            else:
                token = lesk(wordTokens, tokenOrig)
            index = index+1
            #print(token)-
            hyper[token] = []
            hypo[token] = []
            mero[token] = []
            holo[token] = []
            if (token):
                if token.hypernyms():
                    hyper[token] = token.hypernyms()
                if token.hyponyms():
                    hypo[token] = token.hyponyms()
                mero[token] = token.part_meronyms()
                holo[token] = token.part_holonyms()

        return hyper, hypo, mero, holo



if __name__ == "__main__":
    print("starting now...")

    extractFeatures = ExtractFeatures()
    #longestSubsequence in a number
    (extractFeatures.LongestSubsequence("hi hello how are you", "hi there"))

    

    syns = wn.synsets("the")
    #print(syns)
    sent = "the striped bats are hanging on their feet"
    sent1 = "The kitchen is taller than buildings."
    print(word_tokenize(sent))
    print(pos_tag(word_tokenize(sent)))
    print(lesk(['The', 'striped', 'trees', 'are', 'hanging', 'on', 'their', 'feet', 'for', 'game','.'], 'The','s'))
    print(lesk(['Nand', 'will', 'start', 'playing', 'game','.'], 'Nand','n'))



    """print(hyper)
    print("done with hypernyms")
    print(hypo)
    print("done with hyponyms")
    print(mero)
    print("done with meronyms")
    print(holo)
    print("done with holonyms")

    #print(lesk(sent))
    print(pos_tag(word_tokenize(sent)))
    print(pos_tag(word_tokenize(sent1)))
    print("hi")
    print(lesk(['The', 'striped', 'trees', 'are', 'hanging', 'on', 'their', 'feet', 'for', 'game','.'], 'trees','n'))
    print(lesk(['The', 'kicthen', 'is', 'taller', 'than', 'buildings','.'], 'kitchen','n').part_holonyms())
    print(wn.synset('tree.n.01'))
    print(wn.synset('tree.n.01').part_meronyms())
    print(wn.synset('kitchen.n.01').part_meronyms())
    print(lesk(['I', 'went', 'to', 'the','river', 'bank', 'to', 'take', 'water', '.'], 'bank','n'))
    print(lesk(['I', 'can', 'open', 'the','can','.'], 'can','v'))
    print("hi")"""
    # > ['The', 'striped', 'bats', 'are', 'hanging', 'on', 'their', 'feet', 'for', 'best']


    #TA testing tokenize, lemma, pos, dep-parse 
    #this following needs to be uncommented
     
    nlpPipeLine = NlpPipeline()
    nlp = spacy.load("en_core_web_md")

    """sentTest = "TA user input sentence"
    print('-printing all tokens-')
    nlpPipeLine.createTokens(nlp,sentTest)

    print('-printing each lemma-')
    nlpPipeLine.createLemma(nlp, sentTest)

    print('-printing each POS tag-')
    nlpPipeLine.createPOS(nlp, sentTest)

    print('-printing all Dependency parse tree-')
    nlpPipeLine.createDepParse(nlp,sentTest)"""

    #'lemmas' have antonyms
    #'synsets' have hypernyms, hyponyms, meornyms

    data_folder = Path("data/train-set.txt")
    corpusObject = CorpusReader(data_folder)

    #do the nlp pipeline for each parah in corpusObject
    #store in the appropriate HashMap dict
    a = 0
    for corpusParah in corpusObject.corpus:
        #doc1 = nlp(corpusParah.hm1["sent"])
        #doc2 = nlp(corpusParah.hm2["sent"])
        if(a==2):
            break
        hyper, hypo, mero, holo = nlpPipeLine.createAllNyms(corpusParah.hm1["sent"])
        corpusParah.hm1["hyper"] = hyper
        corpusParah.hm1["hypo"] = hypo
        corpusParah.hm1["mero"] = mero
        corpusParah.hm1["holo"] = holo

        hyper, hypo, mero, holo = nlpPipeLine.createAllNyms(corpusParah.hm2["sent"])
        corpusParah.hm2["hyper"] = hyper
        corpusParah.hm2["hypo"] = hypo
        corpusParah.hm2["mero"] = mero
        corpusParah.hm2["holo"] = holo

        #corpusParah.hm1["doc"] = doc1
        #corpusParah.hm2["doc"] = doc2
        a = a+1

    corpusParah1 = corpusObject.corpus[0]
    print(corpusParah1.hm1["sent"])
    doc1 = corpusParah1.hm1["hyper"]
    doc2 = corpusParah1.hm1["hypo"]
    doc3 = corpusParah1.hm1["mero"]
    doc4 = corpusParah1.hm1["holo"]

    print(doc1)
    print(doc2)
    print(doc3)
    print(doc4)

    """corpusParah1 = corpusObject.corpus[0]

    doc1 = corpusParah1.hm1["doc"]
    doc2 = corpusParah1.hm2["doc"]

    for token in doc1:
        print(token)"""

    """for token in doc1:
        print(token.text, token.lemma_, token.pos_, )

    for token in doc2:
        print(token.text, token.lemma_, token.pos_, )"""