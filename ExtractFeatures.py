import difflib
import nltk as dist
from nltk import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
import sklearn.metrics.pairwise as sk
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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

    def longestSubsequence(self, stringSet1, stringSet2):

        s = difflib.SequenceMatcher(isjunk=None, a=stringSet1, b=stringSet2)
        temp = (s.find_longest_match(0, len(stringSet1), 0, len(stringSet2))[2])
        return temp / (len(stringSet1) + len(stringSet2))

        if(len(stringSet1.union(stringSet2)) >222):
            return -1

        hm = {}
        char = 33
        string1 = ""
        string2 = ""
        for word in stringSet1:
            if not word in hm:
                hm[word] = (chr(char))
            string1 = string1 + hm[word]
            char = char + 1

        for word in stringSet2:
            if not word in hm:
                hm[word] = (chr(char))
            string2 = string2 + hm[word]
            char = char + 1

        return (self.lcs(string1, string2)/len(string1+string2))

    #taken from lcs problem of geeksforgeeks
    def lcs(self, X, Y):
        # find the length of the strings
        m = len(X)
        n = len(Y)

        # declaring the array for storing the dp values
        L = [[None] * (n + 1) for i in range(m + 1)]

        """Following steps build L[m + 1][n + 1] in bottom up fashion 
        Note: L[i][j] contains length of LCS of X[0..i-1] 
        and Y[0..j-1]"""
        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0 or j == 0:
                    L[i][j] = 0
                elif X[i - 1] == Y[j - 1]:
                    L[i][j] = L[i - 1][j - 1] + 1
                else:
                    L[i][j] = max(L[i - 1][j], L[i][j - 1])

                    # L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1]
        return L[m][n]
        # end of function lcs

    def jaccardDistance(self, stringSet1, stringSet2):
        #dist.jaccard_distance()
        if(len(stringSet1)==0 & len(stringSet2)==0):
            return -1
        return dist.jaccard_distance(stringSet1, stringSet2)

    def jaccardSimilarity(self, string1, string2):
        #print('doing jaccardSimilarity')
        return self.jaccardDistance(set(word_tokenize(string1)), set(word_tokenize(string2)))

    def lavenshteinDistance(self, doc1, lemmahash1, doc2, lemmahash2):

        #handle strings len> 223

        hm = {}
        char = 33
        string1 = ""
        string2 = ""
        for token in doc1:
            if not lemmahash1[token.text] in hm:
                hm[lemmahash1[token.text]] = (chr(char))
            string1 = string1 + hm[lemmahash1[token.text]]
            char = char + 1

        for token in doc2:
            if not lemmahash2[token.text] in hm:
                hm[lemmahash2[token.text]] = (chr(char))
            string2 = string2 + hm[lemmahash2[token.text]]
            char = char + 1

        #new return
        temp = dist.edit_distance(string1, string2, substitution_cost=1, transpositions=True)
        return temp / (len(string1) + len(string2))

        #print('doing lavenshteinDistance')
        #old return
        temp = dist.edit_distance(stringSet1, stringSet2, substitution_cost=1, transpositions=True)
        return temp / (len(stringSet1) + len(stringSet2))
    
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

    #code is imported from 'https://towardsdatascience.com/overview-of-text-similarity-metrics-3397c4601f50'
    def cosineSimilarities(self, *strs):
        vectors = [t for t in self.get_vectors(*strs)]
        out = 1.0
        for val in cosine_similarity(vectors)[0]:
            out = out * val
        return out

    def get_vectors(self, *strs):
        text = [t for t in strs]
        vectorizer = CountVectorizer(text)
        vectorizer.fit(text)
        return vectorizer.transform(text).toarray()

    def spacySimilarities(self, corpusObject):
        print('doing spacySimilarities')



        lemmaDist = []
        nsubjDist = []
        pobjDist = []
        dobjDist = []
        index = 0
        for corpusParah in corpusObject.corpus:
            doc1 = corpusParah.hm1["doc"]
            doc2 = corpusParah.hm2["doc"]
            nsubj1 = set()
            nsubj2 = set()

            pobj1 = set()
            pobj2 = set()

            dobj1 = set()
            dobj2 = set()

            for token in doc1:
                if(token.dep_ == 'nsubj'):
                    nsubj1.add(token.lemma_)
                if(token.dep_ == 'pobj'):
                    pobj1.add(token.lemma_)
                if(token.dep_ == 'dobj'):
                    dobj1.add(token.lemma_)

            for token in doc2:
                if(token.dep_ == 'nsubj'):
                    nsubj2.add(token.lemma_)
                if(token.dep_ == 'pobj'):
                    pobj2.add(token.lemma_)
                if(token.dep_ == 'dobj'):
                    dobj2.add(token.lemma_)

            lemmaDist.append(self.jaccardDistance(corpusParah.hm1["lemmaset"], corpusParah.hm2["lemmaset"]))
            nsubjDist.append(self.jaccardDistance(nsubj1, nsubj2))

            pobjDist.append(self.jaccardDistance(pobj1, pobj2))
            dobjDist.append(self.jaccardDistance(dobj1, dobj2))
            index = index + 1
        print('done with spacySimilarities')
        return lemmaDist, nsubjDist, pobjDist, dobjDist

    def nGramOverlap(self, doc1, doc2):
        tokens1 = list();
        tokens2 = list();
        for token in doc1:
            tokens1.append(token.lemma_)

        for token in doc2:
            tokens2.append(token.lemma_)

        bigramSet1 = set();
        bigramSet2 = set()
        for index in range(len(tokens1) - 1):
            temp = tokens1[index] + " " + tokens1[index + 1]
            bigramSet1.add(temp)

        for index in range(len(tokens2) - 1):
            temp = tokens2[index] + " " + tokens2[index + 1]
            bigramSet2.add(temp)

        intersectionLengthBigram = len(bigramSet1.intersection(bigramSet2))
        if (intersectionLengthBigram == 0):
            biGramScore = 0
        else:
            biGramSum = (len(bigramSet1) / intersectionLengthBigram) + (len(bigramSet2) / intersectionLengthBigram)
            biGramScore = (2 * (1 / biGramSum))

        triGramSet1 = set();
        triGramSet2 = set()
        for index in range(len(tokens1) - 2):
            temp = tokens1[index] + " " + tokens1[index + 1] + " " + tokens1[index + 2]
            triGramSet1.add(temp)

        for index in range(len(tokens2) - 2):
            temp = tokens2[index] + " " + tokens2[index + 1] + " " + tokens2[index + 2]
            triGramSet2.add(temp)

        intersectionLengthTriGram = len(triGramSet1.intersection(triGramSet2))

        if(intersectionLengthTriGram == 0):
            triGramScore = 0
        else:
            triGramSum = (len(triGramSet1) / intersectionLengthTriGram) + (len(triGramSet2) / intersectionLengthTriGram)
            triGramScore = (2 * (1 / triGramSum))

        return biGramScore, triGramScore