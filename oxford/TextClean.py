# from ChocoNoir import StemNStpw
# Package required: nltk(&nltk.corpus)

class TextClean():
    # line is one stream data(sample)
    # postionOfText is the position of text
    # For example line = ['Text',1], then TextClean(line,0)
    def __init__(self,line,positionOfText):
        self.line = line
        # self.colunmes = positionOfText

    def RemoveStpWords(self):
        # from ChocoNoir.StemNStpw import CleanStw
        self.line[self.colunmes] = CleanStw(self.line[self.colunmes])
        return self


    def Stemming(self):
        # from ChocoNoir.StemNStpw import Stemming
        self.line[self.colunmes] = Stemming(self.line[self.colunmes])
        return self

    # Return a sample with the text cleaned
    def GetLine(self):
        self.RemoveStpWords()
        self.Stemming()
        return self.line

def CleanStw(text):
    from nltk.corpus import stopwords
    import re
    StpWds = stopwords.words('english')
    text = ' '.join([word for word in re.sub('\W|_',' ',text).lower().split() if word not in StpWds])
    return text



def Stemming(text):
    from nltk.stem.snowball import SnowballStemmer
    stemmer = SnowballStemmer("english")
    text = ' '.join([stemmer.stem(word) for word in text.split(" ")])
    return text

def textClean(text):
    import re
    return Stemming(CleanStw(re.sub("\d","",text)))