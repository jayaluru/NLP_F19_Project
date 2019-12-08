
import pandas as pd
from ExtractFeatures import ExtractFeatures as ef
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
import spacy
import pickle as pk



class MachineLearningTasks:
    nlp = spacy.load("en_core_web_md")

    def __init__(self, trainCorpusObject, devCorpusObject):

        spacyTrain = self.addSpacyDoc(trainCorpusObject)
        spacyDev = self.addSpacyDoc(devCorpusObject)

        dfTrain = self.createDF(spacyTrain)
        dfTest = self.createDF(spacyDev)

        dfClass = []
        for corpusParah in trainCorpusObject.corpus:
            dfClass.append(corpusParah.score)

        randForest = rf(n_estimators=201, n_jobs=2, random_state=0)
        supportvm = svm.SVC(decision_function_shape='ovo')
        adaboostClassifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),n_estimators=200)

        #to save the models
        rfp = pk.dumps(randForest)
        sp = pk.dumps(supportvm)
        ap = pk.dumps(adaboostClassifier)

        randForest.fit(dfTrain, dfClass)
        supportvm.fit(dfTrain,dfClass)
        adaboostClassifier.fit(dfTrain,dfClass)

        devrandForest = []
        devsupportvm = []
        devaDaboost = []

        for n in randForest.predict(dfTest):
            devrandForest.append(n)

        for prediction in supportvm.predict(dfTest):
            devsupportvm.append(prediction)

        for prediction in adaboostClassifier.predict(dfTest):
            devaDaboost.append(prediction)


        linenum = 1
        index = 0
        file = open("data/prediction.txt", 'w+')
        file.write("id	Gold Tag\n")
        sum = 0
        acc = 0
        for corpusParah in devCorpusObject.corpus:
            curind = index + 1
            print((str)(curind) + " " + (str)(corpusParah.score) + " " + devrandForest[index] + " " + devsupportvm[index] + " " + devaDaboost[index])
            temp = int(self.maxNumber(devrandForest[index],devsupportvm[index],devaDaboost[index]))
            newLine = "s_" + str(linenum) + "\t" + str(temp)
            sum = sum + abs(int(corpusParah.score) - temp)
            if(int(corpusParah.score) == temp):
                acc = acc + 1
            if(linenum == len(devrandForest)):
                file.write(newLine)
            else:
                file.write(newLine + "\n")
            linenum = linenum + 1
            index = index + 1
        file.close()
        print(sum/1209)
        print(acc)


    def addLemmaList(self, doc):
        lemmaSet = set()
        for token in doc:
            lemmaSet.add(token.lemma_)
        return lemmaSet

    def addLemmaHash(self, doc):
        lemmahash = {}
        for token in doc:
            lemmahash[token.text] = token.lemma_
        return lemmahash

    def addSpacyDoc(self, corpusObject):
        print('creating "doc" from spacy')

        index = 0
        for corpusParah in corpusObject.corpus:
            print(index)
            sent1 = corpusParah.hm1["sent"]
            sent2 = corpusParah.hm2["sent"]
            doc1 = self.nlp(sent1)
            doc2 = self.nlp(sent2)
            corpusParah.hm1["doc"] = doc1
            lemmaset = self.addLemmaList(doc1)
            lemmahash = self.addLemmaHash(doc1)
            corpusParah.hm1["lemmaset"] = lemmaset
            corpusParah.hm1["lemmahash"] = lemmahash

            corpusParah.hm2["doc"] = doc2
            lemmaset = self.addLemmaList(doc2)
            lemmahash = self.addLemmaHash(doc2)
            corpusParah.hm2["lemmaset"] = lemmaset
            corpusParah.hm2["lemmahash"] = lemmahash
            index = index + 1

        return corpusObject

    def lemmaString(self, doc):
        outString = ""
        index = 0
        for token in doc:
            if (index == len(doc)):
                outString = outString + token.lemma_
            else:
                outString = outString + token.lemma_ + " "
        return outString


    def createDF(self, corpusObject):
        df = pd.DataFrame(columns=['ls', 'js', 'ld', 'npos', 'vpos', 'apos', 'rpos', 'lemmaDist',
                                   'nsubjDist', 'pobjDist', 'dobjDist', 'cs', 'bigram', 'trigram',
                                   'nsubSimilarity', 'pobjSimilarity', 'dobjSimilarity'])
        """df = pd.DataFrame(columns=['ls', 'js', 'ld', 'npos', 'vpos', 'apos', 'rpos', 'lemmaDist',
                                   'nsubjDist', 'pobjDist', 'dobjDist', 'cs', 'bigram', 'trigram'])"""
        index = 0
        efObject = ef()
        lemmaDist, nsubjDist, pobjDist, dobjDist = efObject.spacySimilarities(corpusObject)
        nsubSimilarity, pobjSimilarity, dobjSimilarity = efObject.wordSimilarity(corpusObject)
        print('extracting features')
        for corpusParah in corpusObject.corpus:
            sent1 = self.lemmaString(corpusParah.hm1["doc"])
            sent2 = self.lemmaString(corpusParah.hm2["doc"])
            ls = efObject.longestSubsequence(corpusParah.hm1["doc"], corpusParah.hm1["lemmahash"], corpusParah.hm2["doc"],  corpusParah.hm2["lemmahash"])
            js = efObject.jaccardDistance(corpusParah.hm1["lemmaset"], corpusParah.hm2["lemmaset"])
            ld = efObject.lavenshteinDistance(corpusParah.hm1["doc"], corpusParah.hm1["lemmahash"], corpusParah.hm2["doc"],  corpusParah.hm2["lemmahash"])
            cs = efObject.cosineSimilarities(sent1, sent2)
            npos, vpos, apos, rpos = efObject.posFeatures(sent1, sent2)
            bigram, trigram = efObject.nGramOverlap(corpusParah.hm1["doc"], corpusParah.hm2["doc"])
            df.loc[index] = [ls, js, ld, npos, vpos, apos, rpos, lemmaDist[index], nsubjDist[index],
                             pobjDist[index], dobjDist[index], cs, bigram, trigram,
                             nsubSimilarity[index], pobjSimilarity[index], dobjSimilarity[index]]
            """df.loc[index] = [ls, js, ld, npos, vpos, apos, rpos, lemmaDist[index], nsubjDist[index],
                             pobjDist[index], dobjDist[index], cs, bigram, trigram]"""
            index = index + 1
        print(df.head())
        return df


    def maxNumber(self,num1,num2,num3):
        if num1==num2:
            return num1
        elif num2==num3:
            return num2
        elif num1==num3:
            return num1
        return num1