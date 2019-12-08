import spacy
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
import re
from nltk import word_tokenize
from nltk import pos_tag
from pathlib import Path
from CorpusReader import CorpusReader
from MachineLearningTasks import MachineLearningTasks
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
        hyperd = {}
        hypod = {}
        merod = {}
        holod = {}
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
                print("")
                print(tokenOrig)
                print(token)
                if token.hypernyms():
                    #hyper.append(token.hypernyms())
                    hyperd[token] = token.hypernyms()
                    print("hypernyms")
                    print(token.hypernyms())
                    print("")
                else:
                    print("there are no hypernyms")
                if token.hyponyms():
                    #hypo[token] = token.hyponyms()
                    hypod[token] = token.hyponyms()
                    print("hyponyms")
                    print(token.hyponyms())
                    print("")
                else:
                    print("there are no hyponyms")
                #mero[token] = token.part_meronyms()
                merod[token] = token.part_meronyms()
                print("meronyms")
                print(token.part_meronyms())
                print("")
                #holo[token] = token.part_holonyms()
                holod[token] = token.part_holonyms()
                print("holonyms")
                print(token.part_holonyms())
                print("")

        return hyperd, hypod, merod, holod



if __name__ == "__main__":
    print("starting now. loading spacy")

    #TA testing tokenize, lemma, pos, dep-parse 
    #this following needs to be uncommented

    '''nlpPipeLine = NlpPipeline()
    nlp = spacy.load("en_core_web_md")

    sentTest = "TA user input sentence"
    sentTest = "The secretariat is expected to run tomorrow"
    print("start of task 2")
    print('')
    print('-printing all tokens-')
    nlpPipeLine.createTokens(nlp,sentTest)

    print('')
    print('-printing each lemma-')
    nlpPipeLine.createLemma(nlp, sentTest)

    print('')
    print('-printing each POS tag-')
    nlpPipeLine.createPOS(nlp, sentTest)

    print('')
    print('-printing all Dependency parse tree-')
    nlpPipeLine.createDepParse(nlp,sentTest)

    print('')
    print('-printing all Dependency parse tree-')
    hyper, hypo, mero, holo = nlpPipeLine.createAllNyms(sentTest)


    print("end of task 2")'''
    data_folder_train = Path("data/train-set.txt")
    trainCorpusObject = CorpusReader(data_folder_train)

    data_folder_test = Path("data/test-set.txt")
    devCorpusObject = CorpusReader(data_folder_test)


    mlObject = MachineLearningTasks(trainCorpusObject, devCorpusObject)

    #do the nlp pipeline for each parah in corpusObject
    #store in the appropriate HashMap dict

    """a = 0
    for corpusParah in trainCorpusObject.corpus:
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
        a = a+1"""

    """corpusParah1 = corpusObject.corpus[0]
    print(corpusParah1.hm1["sent"])
    doc1 = corpusParah1.hm1["hyper"]
    doc2 = corpusParah1.hm1["hypo"]
    doc3 = corpusParah1.hm1["mero"]
    doc4 = corpusParah1.hm1["holo"]

    print(doc1)
    print(doc2)
    print(doc3)
    print(doc4)"""

    """corpusParah1 = corpusObject.corpus[0]

    doc1 = corpusParah1.hm1["doc"]
    doc2 = corpusParah1.hm2["doc"]

    for token in doc1:
        print(token)"""

    """for token in doc1:
        print(token.text, token.lemma_, token.pos_, )

    for token in doc2:
        print(token.text, token.lemma_, token.pos_, )"""