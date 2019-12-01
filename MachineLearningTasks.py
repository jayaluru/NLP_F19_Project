import csv
import pandas as pd
from ExtractFeatures import ExtractFeatures as ef
from sklearn.ensemble import RandomForestClassifier as rf


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
        index = 0
        sum = 0
        for corpusParah in devCorpusObject.corpus:
            sum = sum + abs(int(corpusParah.score) - int(devVal[index]))
            index = index + 1

        print(sum/1209)


    def createDF(self, corpusObject):
        df = pd.DataFrame(columns=['ls', 'js', 'ld', 'npos', 'vpos', 'apos', 'rpos', 'lemmaDist', 'nsubjDist', 'pobjDist', 'dobjDist'])
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
            npos, vpos, apos, rpos = efObject.posFeatures(sent1, sent2)
            #lemmaDist, nsubjDist = efObject.spacySimilarities(sent1, sent2)
            df.loc[index] = [ls, js, ld, npos, vpos, apos, rpos, lemmaDist[index], nsubjDist[index], pobjDist[index], dobjDist[index]]
            index = index + 1
        print(df.head())
        return df


