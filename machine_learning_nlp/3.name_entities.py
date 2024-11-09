sentences ="The Eiffle Tower was built from 1887 to 1889 by French engineer Gustave Eiffel , whose company specialized in building metal frameworks and structures."

import nltk 
from nltk.tokenize import word_tokenize
nltk.download('maxent_ne_chunker_tab')

word_tokenization = word_tokenize(sentences)  # word tokenization implemented

print("**** word tokenization is printing ")
print(word_tokenization)
print()

tag_elements =nltk.pos_tag(word_tokenization)                   # part of speech implemeneted
print("**** part of Speech is printing ")
print(tag_elements)
print()

words_chunks =nltk.ne_chunk(tag_elements).draw()
print("**** chunks is printing ")
print(words_chunks)
print()