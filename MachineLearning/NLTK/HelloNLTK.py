import nltk
from nltk.tokenize import sent_tokenize , word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

data = "this is just to demonstrate the use of nltk (natural language tool kit) in python and how to leverage its capabilities in computations. " \
       "Do you have any plans to learn it ? I can say this would be a good learning experience."
print(word_tokenize(data))
print(sent_tokenize(data))

# removing few words which are stopwords
words = word_tokenize(data)
wordsFiltered = []
stop_words = set(stopwords.words('english'))
for word in words:
    if word not in stop_words:
        wordsFiltered.append(word)

print(wordsFiltered)

#to identify the stem words from a given data
ps = PorterStemmer()
for word in words:
    print(ps.stem(word))

#NLTK – speech tagging
#The below example  can automatically tag words with a corresponding vocab.
sentences = sent_tokenize(data)
for sent in sentences:
    print(nltk.pos_tag(nltk.word_tokenize(sent)))

