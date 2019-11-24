import spacy
from nltk.corpus import wordnet
from nltk.wsd import lesk

from nltk import word_tokenize
from nltk import pos_tag
from pathlib import Path
from CorpusReader import CorpusReader
from spacy.lang.en import English



class NlpPipeline:

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


if __name__ == "__main__":
    print("starting now...")

    """sent = "the striped bats are hanging on their feet"
    sent1 = "i can open the can"
    #print(lesk(sent))
    print(pos_tag(word_tokenize(sent)))
    print(pos_tag(word_tokenize(sent1)))
    print("hi")
    print(lesk(['The', 'striped', 'bats', 'are', 'hanging', 'on', 'their', 'feet', 'for', 'game','.'], 'bats'))
    print(lesk(['I', 'went', 'to', 'the','river', 'bank', 'to', 'take', 'water', '.'], 'bank'))
    print(lesk(['I', 'can', 'open', 'the','can','.'], 'can','v'))
    print("hi")
    # > ['The', 'striped', 'bats', 'are', 'hanging', 'on', 'their', 'feet', 'for', 'best']"""

    """
    #TA testing tokenize, lemma, pos, dep-parse 
    #this following needs to be uncommented
     
    nlpPipeLine = NlpPipeline()
    nlp = spacy.load("en_core_web_md")

    sentTest = "TA user input sentence"
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

    synonyms = []
    antonyms = []
    hypernyms = []
    hyponyms = []
    meronyms = []
    holonyms = []

    syns = wordnet.synsets("very")
    dog = wordnet.synset('dog.n.01')
    print(dog.pos())

    #print(dog.hypernyms())
    print(syns)
    #print(wordnet.synset('kitchen.n.01').part_holonyms())
    #print(lesk(['I', 'went', 'to', 'the', 'bank', 'to', 'deposit', 'money', '.'], 'bank', 'n'))

    for syn in syns:
        if syn.hypernyms():
            hypernyms.append(syn.hypernyms()[0].name())
        if syn.hyponyms():
            hyponyms.append(syn.hyponyms()[0].name())
        """if syn.meronyms():
            meronyms.append(syn.meronyms()[0].name())"""
        """if syn.holonyms():
            holonyms.append(syn.holonyms()[0].name())"""

        """for l in syn.lemmas():
            if True:
                synonyms.append(l.name())
                #all lemmas are here considered synonyms
            if l.antonyms():
                antonyms.append(l.antonyms()[0].name())"""

    #print(syn.lemmas())
    #print(synonyms)

    #print(antonyms)
    print(hypernyms)
    print(hyponyms)
    #print(meronyms)
    #print(holonyms)


    """data_folder = Path("data/train-set.txt")
    corpusObject = CorpusReader(data_folder)

    #do the nlp pipeline for each parah in corpusObject
    #store in the appropriate HashMap dict 
    for corpusParah in corpusObject.corpus:
        doc1 = nlp(corpusParah.hm1["sent"])
        doc2 = nlp(corpusParah.hm2["sent"])

        corpusParah.hm1["doc"] = doc1
        corpusParah.hm2["doc"] = doc2"""





    """corpusParah1 = corpusObject.corpus[0]

    doc1 = corpusParah1.hm1["doc"]
    doc2 = corpusParah1.hm2["doc"]

    for token in doc1:
        print(token)"""

    """for token in doc1:
        print(token.text, token.lemma_, token.pos_, )

    for token in doc2:
        print(token.text, token.lemma_, token.pos_, )"""