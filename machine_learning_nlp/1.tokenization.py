corpus ="""
In publishing and graphic design, Lorem ipsum (/ˌlɔː.rəm ˈɪp.səm/ LOR-əm IP-səm) is a placeholder text commonly used to demonstrate the visual form of a document or a typeface without relying on meaningful content. Lorem ipsum may be used as a placeholder before the final copy is available. It is also used to temporarily replace text in a process called greeking, which allows designers to consider the form of a webpage or publication, without the meaning of the text influencing the design.
"""

# 1 . tokenization paragraph into sentence
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

sentence_tokenize_text =sent_tokenize(corpus)               # implement sentence tokenizer
for sentences in sentence_tokenize_text:
    sentence_word_tokenization = word_tokenize(sentences)
    print('1. ---------------------- Sentence tokenization is printing ')
    print(sentence_word_tokenization)
    print()



# 2 . user word tokenization 
word_tokenize_text = word_tokenize(corpus)
print("2. ---------------------------- word tokenization is printing ")
print(word_tokenize_text)
print()


# 3. treebank word tokenizer

from nltk.tokenize import TreebankWordTokenizer
tree_tokenizer= TreebankWordTokenizer()
tokenizer_text =tree_tokenizer.tokenize(corpus)
print()
print("3. ---------------------------- Tree tokenization is printing ")
print(tokenizer_text)
