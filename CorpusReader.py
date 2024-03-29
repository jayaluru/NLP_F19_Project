import re

class CorpusReader:
    def __init__(self, train):
        file = open(train, 'rb')
        file_string_byte = file.read()
        file_string = file_string_byte.decode('utf-8')
        tokens = file_string.split("\n")
        self.corpus = []
        index = 0
        for token in tokens[1:]:
            #print(index)
            paragh = ParaghData(token.rstrip())
            self.corpus.append(paragh)
            index = index +1


class ParaghData:
    def __init__(self, paragh):
        paragh = re.sub(r'["\']', '', paragh)
        splitted = re.split(r'\t+', paragh)
        #print(len(splitted))
        self.hm1 = {}
        self.hm2 = {}
        self.hm1["sent"] = splitted[1]
        self.hm2["sent"] = splitted[2]

        if(len(splitted)==4):
            self.score = splitted[3]
        else:
            self.score = 0
        self.id = splitted[0]

