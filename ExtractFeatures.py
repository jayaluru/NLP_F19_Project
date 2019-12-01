import difflib
import spacy
import nltk as dist
from nltk import word_tokenize
from nltk import pos_tag


class ExtractFeatures:

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

    def longestSubsequence(self, string1, string2):
        #print('doing longestSubsequence')
        s = difflib.SequenceMatcher(isjunk=None, a=string1, b=string2)
        temp = (s.find_longest_match(0,len(string1),0,len(string2))[2])
        return temp/(len(string1)+len(string2))

    def jaccardDistance(self, stringSet1, stringSet2):
        #dist.jaccard_distance()
        if(len(stringSet1)==0 & len(stringSet2)==0):
            return -1
        return dist.jaccard_distance(stringSet1, stringSet2)

    def jaccardSimilarity(self, string1, string2):
        #print('doing jaccardSimilarity')
        return self.jaccardDistance(set(word_tokenize(string1)), set(word_tokenize(string2)))

    def lavenshteinDistance(self, string1, string2):
        #print('doing lavenshteinDistance')
        temp = dist.edit_distance(string1, string2, substitution_cost=1, transpositions=True)
        return temp / (len(string1) + len(string2))
    
    def posFeatures(self, string1, string2):
        #print('doing posFeatures')
        stringSet1 = set(pos_tag(word_tokenize(string1)))
        stringSet2 = set(pos_tag(word_tokenize(string2)))
        n = self.posOverlapJaccard(stringSet1, stringSet2, 'n')
        v = self.posOverlapJaccard(stringSet1, stringSet2, 'v')
        a = self.posOverlapJaccard(stringSet1, stringSet2, 'a')
        r = self.posOverlapJaccard(stringSet1, stringSet2, 'r')
        return n, v, a, r

    def posOverlapJaccard(self, stringSet1, stringSet2, pos):
        newStringSet1 = set()
        for val in stringSet1:
            if (val[1] in self.wordnet_tag_map):
                if (self.wordnet_tag_map[val[1]] == pos):
                    newStringSet1.add(val[0])

        newStringSet2 = set()
        for val in stringSet2:
            if (val[1] in self.wordnet_tag_map):
                if (self.wordnet_tag_map[val[1]] == pos):
                    newStringSet2.add(val[0])

        #print(newStringSet1)
        #print(newStringSet2)
        return self.jaccardDistance(newStringSet1, newStringSet2)

    def spacySimilarities(self, corpusObject):
        print('doing spacySimilarities')
        nlp = spacy.load("en_core_web_md")
        lemmaSet1 = set()
        lemmaSet2 = set()
        nsubj1 = set()
        nsubj2 = set()

        pobj1 = set()
        pobj2 = set()

        dobj1 = set()
        dobj2 = set()

        lemmaDist = []
        nsubjDist = []
        pobjDist = []
        dobjDist = []
        index = 0
        for corpusParah in corpusObject.corpus:
            print(index)
            sent1 = corpusParah.hm1["sent"]
            sent2 = corpusParah.hm2["sent"]
            doc1 = nlp(sent1)
            doc2 = nlp(sent2)

            for token in doc1:
                lemmaSet1.add(token.lemma_)
                if(token.dep_ == 'nsubj'):
                    nsubj1.add(token.lemma_)
                if(token.dep_ == 'pobj'):
                    pobj1.add(token.lemma_)
                if(token.dep_ == 'dobj'):
                    dobj1.add(token.lemma_)

            for token in doc2:
                lemmaSet2.add(token.lemma_)
                if(token.dep_ == 'nsubj'):
                    nsubj2.add(token.lemma_)
                if(token.dep_ == 'pobj'):
                    pobj2.add(token.lemma_)
                if(token.dep_ == 'dobj'):
                    dobj2.add(token.lemma_)

            lemmaDist.append(self.jaccardDistance(lemmaSet1, lemmaSet2)**4)
            nsubjDist.append(self.jaccardDistance(nsubj1, nsubj2))

            pobjDist.append(self.jaccardDistance(pobj1, pobj2))
            dobjDist.append(self.jaccardDistance(dobj1, dobj2))
            index = index + 1
        return lemmaDist, nsubjDist, pobjDist, dobjDist