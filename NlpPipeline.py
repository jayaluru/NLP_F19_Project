import spacy
from pathlib import Path
from CorpusReader import CorpusReader
from spacy.lang.en import English



class NlpPipeline:
    def _init_(self):
        print("Hi")


    def createTokens(self,tokenizer,sentence):
        tokens = tokenizer(sentence)

    def createDepParse(self,nlpdep,sentence):
        doc = nlpdep(sentence)
        for token in doc:
            print(token.text, token.dep_, token.head.text, token.head.pos_,
                  [child for child in token.children])

    def createLemma(self,nlpdep,sentence):
        doc = nlpdep(sentence)
        for token in doc:
            print(token.text,token.lemma_)

    def createPOS(self,nlpdep,sentence):
        doc = nlpdep(sentence)
        for token in doc:
            print(token.text, token.pos_)



if __name__ == "__main__":

    print("hello")
    nlpPipeLine = NlpPipeline()

    data_folder = Path("data/train-set.txt")
    corpusObject = CorpusReader(data_folder)



    nlp = spacy.load("en_core_web_md")


    for corpusParah in corpusObject.corpus:
        doc1 = nlp(corpusParah.hm1["sent"])
        doc2 = nlp(corpusParah.hm2["sent"])

        corpusParah.hm1["doc"] = doc1
        corpusParah.hm2["doc"] = doc2

    corpusParah1 = corpusObject.corpus[0]

    doc1 = corpusParah1.hm1["doc"]
    doc2 = corpusParah1.hm2["doc"]

    for token in doc1:
        print(token.text, token.lemma_, token.pos_, )

    for token in doc2:
        print(token.text, token.lemma_, token.pos_, )