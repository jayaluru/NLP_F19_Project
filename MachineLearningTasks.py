import csv
import pandas as pd
from ExtractFeatures import ExtractFeatures as ef
from sklearn.ensemble import RandomForestClassifier as rf


class MachineLearningTasks:

    def __init__(self, trainCorpusObject, testCorpusObject):
        dfTrain = self.createDF(trainCorpusObject)
        dfTest = self.createDF(testCorpusObject)

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
        for corpusParah in testCorpusObject.corpus:
            sum = sum + abs(int(corpusParah.score) - int(devVal[index]))
            index = index + 1

        print(sum/1209)


    def createDF(self, corpusObject):
        df = pd.DataFrame(columns=['ls', 'js', 'ld', 'npos', 'vpos', 'apos', 'rpos'])
        index = 0
        for corpusParah in corpusObject.corpus:
            sent1 = corpusParah.hm1["sent"]
            sent2 = corpusParah.hm2["sent"]
            efObject = ef()
            ls = efObject.longestSubsequence(sent1, sent2)
            js = efObject.jaccardSimilarity(sent1, sent2)
            ld = efObject.lavenshteinDistance(sent1, sent2)
            npos, vpos, apos, rpos = efObject.posFeatures(sent1, sent2)
            df.loc[index] = [ls, js, ld, npos, vpos, apos, rpos]
            index = index + 1
        print(df.head())
        return df

