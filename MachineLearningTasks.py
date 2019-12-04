import csv
import pandas as pd
from ExtractFeatures import ExtractFeatures as ef
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from random import randint
import spacy



class MachineLearningTasks:

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
        randForest.fit(dfTrain, dfClass)
        supportvm.fit(dfTrain,dfClass)
        adaboostClassifier.fit(dfTrain,dfClass)
        devrandForest = []
        devsupportvm = []
        devaDaboost = []
        #clf.predict(dfTest)

        for n in randForest.predict(dfTest):
            devrandForest.append(n)
            #print(n)

        for prediction in supportvm.predict(dfTest):
            devsupportvm.append(prediction)

        for prediction in adaboostClassifier.predict(dfTest):
            devaDaboost.append(prediction)


        devRel = []
        linenum = 1
        index = 0
        #sum = 0
        #sum2 = 0
        #print("Original-Projected")
        file = open("data/prediction.txt", 'w+')
        file.write("id	Gold Tag\n")
        for corpusParah in devCorpusObject.corpus:
            #rd = randint(1,5)
            #sum = sum + abs(float(corpusParah.score) - float(devVal[index]))
            #sum2 = sum2 + abs(float(corpusParah.score) - float(rd))
            print(corpusParah.score + " " + devrandForest[index] + " " + devsupportvm[index] + " " + devaDaboost[index])
            newLine = "s_" + str(linenum) + "\t" + str(self.maxNumber(devrandForest[index],devsupportvm[index],devaDaboost[index]))
            if(linenum == len(devrandForest)):
                file.write(newLine)
            else:
                file.write(newLine + "\n")
            #print("hi")
            linenum = linenum + 1
            index = index + 1
        file.close()



        #print(sum/1209)
        #print(sum2/1209)

    def addSpacyDoc(self, corpusObject):
        print('creating "doc" from spacy')
        nlp = spacy.load("en_core_web_md")
        index = 0

        for corpusParah in corpusObject.corpus:
            print(index)
            sent1 = corpusParah.hm1["sent"]
            sent2 = corpusParah.hm2["sent"]
            doc1 = nlp(sent1)
            doc2 = nlp(sent2)
            corpusParah.hm1["doc"] = doc1
            corpusParah.hm2["doc"] = doc2
            index = index +1

    def createDF(self, corpusObject):
        df = pd.DataFrame(columns=['ls', 'js', 'ld', 'npos', 'vpos', 'apos', 'rpos', 'lemmaDist', 'nsubjDist', 'pobjDist', 'dobjDist', 'cs'])
        #df = pd.DataFrame(columns=['ls', 'js', 'ld', 'npos', 'vpos', 'apos', 'rpos', 'lemmaDist', 'cs'])
        index = 0
        efObject = ef()
        lemmaDist, nsubjDist, pobjDist, dobjDist = efObject.spacySimilarities(corpusObject)
        print('extracting features')
        for corpusParah in corpusObject.corpus:
            sent1 = corpusParah.hm1["sent"]
            sent2 = corpusParah.hm2["sent"]
            ls = efObject.longestSubsequence(sent1, sent2)
            js = efObject.jaccardSimilarity(sent1, sent2)
            ld = efObject.lavenshteinDistance(sent1, sent2)
            cs = efObject.cosineSimilarities(sent1, sent2)
            npos, vpos, apos, rpos = efObject.posFeatures(sent1, sent2)
            df.loc[index] = [ls, js, ld, npos, vpos, apos, rpos, lemmaDist[index], nsubjDist[index], pobjDist[index], dobjDist[index], cs]
            #df.loc[index] = [ls, js, ld, npos, vpos, apos, rpos, lemmaDist[index], cs]
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