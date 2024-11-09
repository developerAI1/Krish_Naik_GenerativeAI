paragraph ="""
Kalam believed that India should stand up to the world as a developed nation, and that it should be strong economically and militarily. He also believed that India should protect and nurture the freedom it gained in the war of Independence in 1857. Kalam envisioned India as a developed nation by 2020, with a reduced rural-urban divide, equitable access to energy and water, and a strong education system. He also envisioned a nation where people of different faiths could live together and work together for the nation's development. Kalam spoke about the importance of having a strong vision, and how young people can discover their potential and become greater than they ever dreamed. He also spoke about the importance of teachers, and how they can be role models for their students. I believe that India got its first vision of this in 1857, when we started the war of Independence. It is this freedom that we must protect and nurture and build on. If we are not free, no one will respect us. My second vision for India's DEVELOPMENT, For fifty years we have been A developing nation.
"""

import nltk
from nltk.tokenize import sent_tokenize , word_tokenize
from nltk.corpus import stopwords
# apply sentence tokenizer on paragraph text
sentences =sent_tokenize(paragraph)

for i in range(len(sentences)):
    # implement word tokenizer on senetences.
    words =word_tokenize(sentences[i])
    
    # implement stop words on words 
    words=  [word for word in words if word not in set(stopwords.words('english'))]
    # sentences[i]  =" ".join(words)
    
    # implement POS tag on words list
    pos_tag =nltk.pos_tag(words)
    print(pos_tag)