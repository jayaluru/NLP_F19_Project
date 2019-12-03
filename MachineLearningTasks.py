import csv
import pandas as pd
from ExtractFeatures import ExtractFeatures as ef
from sklearn.ensemble import RandomForestClassifier as rf
from random import randint


class MachineLearningTasks:

    def __init__(self, trainCorpusObject, devCorpusObject):
        dfTrain = self.createDF(trainCorpusObject)
        dfTest = self.createDF(devCorpusObject)

        dfClass = []
        for corpusParah in trainCorpusObject.corpus:
            dfClass.append(corpusParah.score)

        clf = rf(n_estimators=201, n_jobs=2, random_state=0)
        clf.fit(dfTrain, dfClass)
        devVal = []
        #clf.predict(dfTest)

        for n in clf.predict(dfTest):
            devVal.append(n)
            #print(n)

        devRel = []
        linenum = 1
        #sum = 0
        #sum2 = 0
        #print("Original-Projected")
        file = open("data/prediction.txt", 'w+')
        file.write("id	Gold Tag\n")
        for index in devVal:
            #rd = randint(1,5)
            #sum = sum + abs(float(corpusParah.score) - float(devVal[index]))
            #sum2 = sum2 + abs(float(corpusParah.score) - float(rd))
            #print(corpusParah.score + " " + devVal[index])
            newLine = "s_" + str(linenum) + " " + str(index)
            linenum = linenum +1
            if(linenum == len(devVal)):
                (file.write(newLine + "\n"))
            else:
                file.write(newLine)
            #print("hi")

        file.close()



        #print(sum/1209)
        #print(sum2/1209)


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


