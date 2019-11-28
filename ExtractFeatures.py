import difflib

class ExtractFeatures:

    def LongestSubsequence(self, string1, string2):
        s = difflib.SequenceMatcher(isjunk=None, a=string1, b=string2)
        return (s.find_longest_match(0,len(string1),0,len(string2))[2])