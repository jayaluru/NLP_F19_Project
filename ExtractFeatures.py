import difflib
import nltk as dist

class ExtractFeatures:

    def longestSubsequence(self, string1, string2):
        s = difflib.SequenceMatcher(isjunk=None, a=string1, b=string2)
        return (s.find_longest_match(0,len(string1),0,len(string2))[2])

    def jaccardDistance(self, stringSet1, stringSet2):
        #dist.jaccard_distance()
        return dist.jaccard_distance(stringSet1, stringSet2)

    def lavenshteinDistance(self, string1, string2):
        return dist.edit_distance(string1, string2, substitution_cost=1, transpositions=True)