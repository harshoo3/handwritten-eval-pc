# !python -m spacy download en_core_web_lg
# pip install yake
import regex as re
import yake
import math
from scipy import spatial

import spacy
from spacy.lang.en import English
import en_core_web_lg
nlp = en_core_web_lg.load()
from spacy.matcher import PhraseMatcher
phrase_matcher = PhraseMatcher(nlp.vocab)
# nlp = spacy.load('en_core_web_lg')

import nltk
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

#function to extract keywords
def extract_nouns_adj(lines):
  tokenized = nltk.word_tokenize(lines)
  list_word_tags = nltk.pos_tag(tokenized)
  # function to test if something is a noun or adj
  is_noun_adj = lambda pos: pos[:2] == 'NN' or pos[:2] == 'JJ' or pos[:2] == 'VB'
  # do the nlp stuff
  # tokenized = nltk.word_tokenize(lines)
  nouns_adjs = [(word,pos) for (word, pos) in list_word_tags if is_noun_adj(pos)] 
  # print(nouns_adjs)
  return nouns_adjs

def Regex(article_text):   
  processed_article = article_text.lower()  
  processed_article = re.sub('[^a-zA-Z]', ' ', processed_article )  
  processed_article = re.sub(r'\s+', ' ', processed_article)
  # print(processed_article)
  return processed_article

def text_processing(answer):
  answer=[i for i in answer.split('.')if i != '']
  answer = [Regex(line) for line in answer]
  print(answer)
  return answer

from scipy import spatial
def createKeywordsVectors(keyword, nlp):
    doc = nlp(keyword)  # convert to document object

    return doc.vector

# method to find cosine similarity
def cosineSimilarity(vect1, vect2):
    # return cosine distance
    return 1 - spatial.distance.cosine(vect1, vect2)

# method to find similar words
def getSimilarWords(keyword_and_pos, nlp):
    wordnet_synonyms = []
    similarity_list = []
    keyword,og_tag = keyword_and_pos 
    keyword_vector = createKeywordsVectors(keyword, nlp)

    for syn in wordnet.synsets(keyword):
      for l in syn.lemmas():
        word = l.name()
        if(word.find('_')==-1):
          wordnet_synonyms.append(nlp.vocab[word])

    for tokens in wordnet_synonyms:
        try:
            if (tokens.has_vector):
              if (tokens.is_lower):
                  if (tokens.is_alpha):
                      similarity_list.append((tokens, cosineSimilarity(keyword_vector, tokens.vector)))
        except:
          pass

    similarity_list = sorted(similarity_list, key=lambda item: -item[1])
    similarity_list = list(set(similarity_list))

    top_similar_words = [item[0].text for item in similarity_list]
    # is_noun_adj = lambda pos: pos[:2] == og_tag
    # top_similar_words = [word for (word,tag) in [nltk.pos_tag([word])[0] for word in top_similar_words] if is_noun_adj(tag)]
    top_similar_words = top_similar_words[:5]

    top_similar_words.append(keyword)
  
    for token in nlp(keyword):
        top_similar_words.insert(0, token.lemma_)
  
    top_similar_words = list(set(top_similar_words))

    return top_similar_words

def extract_keywords(text):
  kw_extractor = yake.KeywordExtractor()
  language = "en"
  max_ngram_size = 1
  deduplication_threshold = 0.5
  numOfKeywords = 100
  custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=None)
  keywords = custom_kw_extractor.extract_keywords(text)
  keywords_list = [word for(word,_) in keywords]

  # print(keywords_list)
  return keywords_list

def keyword_count(test_list):
  noun_count = 0
  adj_count = 0
  verb_count = 0
  for (text,tag) in test_list:
    tag=tag[:2]
    # print(str(text)+': '+tag)
    if tag == 'NN':
      noun_count+=1
    elif tag == 'JJ':
      adj_count+=1
    else:
      verb_count+=1
  # print(noun_count+':'+adj_count+':'+verb_count)
  return noun_count,adj_count,verb_count

def match_keywords(student_answer,teacher_answer):
  lemmatizer = WordNetLemmatizer()
  new_teacher_answer = ' '.join([lemmatizer.lemmatize(t) for t in word_tokenize(Regex(teacher_answer)) if t not in stop_words])
  test_list = list(set(extract_keywords(new_teacher_answer)))
  test_list = nltk.pos_tag(test_list)
  noun_count,adj_count,verb_count = keyword_count(test_list)
  print("Keywords found in teacher's answer:")
  print(test_list)
  
  new_test_list = []
  print("Similar word generation for the keywords:")
  for words in test_list:
    # print(words)
    sim_words = getSimilarWords(words,nlp)
    print(str(words)+':'+str(sim_words))
    # print(str(words))
    [new_test_list.append((w,words[1])) for w in sim_words]
  for words in new_test_list:
    test_list.append(words)
  # print(test_list)
  test_list = list(set(test_list))
  final_test_list = [(w,tag) for (w,tag) in test_list if w not in stop_words] 
  patterns = [nlp(text) for (text,tag) in final_test_list]
  Dict={}
  # print(final_test_list)
  for (w,tag) in final_test_list:
    Dict[w]=tag
  # noun_count,adj_count,verb_count= keyword_count(final_test_list)
  # print(str(noun_count)+":"+str(adj_count)+":"+str(verb_count))
  phrase_matcher.add('PhraseMatcher', None, *patterns)
  sentence = nlp (student_answer)

  matched_phrases = phrase_matcher(sentence)
  for match_id, start, end in matched_phrases:
    string_id = nlp.vocab.strings[match_id]  
    span = sentence[start:end]                   
    print(str(span.text)+" : word location in student's answer: "+str(start))

  return matched_phrases,Dict,noun_count,adj_count,verb_count

def marks_eval(student_answer,keyword_match_list,Dict,noun_count,adj_count,verb_count,marks):
  n_count=0
  a_count=0
  v_count=0
  noun_check = lambda x:x[:2]=='NN'
  adj_check = lambda x:x[:2]=='JJ'
  verb_check = lambda x:x[:2]=='VB'
  for word in keyword_match_list:
    # print(Dict[word])
    if noun_check(Dict[word]):
      n_count+=1
    elif verb_check(Dict[word]):
      v_count+=1
    elif adj_check(Dict[word]):
      a_count+=1
  print("Total Nouns: Adj: Verbs")
  print(str(noun_count)+":"+str(adj_count)+":"+str(verb_count))
  print("Matched Nouns: Adj: Verbs")
  print(str(n_count)+":"+str(a_count)+":"+str(v_count))
  weight = 1/(float)(2*noun_count + adj_count + verb_count)
  fraction = (2*n_count + a_count + v_count)*weight
  score = marks * fraction
  score = 2*(score)
  score = math.ceil(score)
  score = float(score)/2
  print("fraction matched: "+str(fraction))
  print("Score: "+str(score)+"/"+str(marks))
  

def nlp_main(teacher_answer:str, student_answer:str,marks:int):
# def main():
    # student_answer = "Confidentiality: Only the sender and the receiver should be able to understand the contents of the transmitted message. The message may be encrypted due to hackers. This is the most commonly perceived meaning of secure communication. Authentication: Both sender and receiver should be able to confirm the identity of the other party involved in the communication i.e to cofirm that the other party is indeed who or what they claim to be. Message integrity and non-repudiation: Even if the sender and receiver are authenticated, they ensure that the context of their communication is not altered. Message integrity can be ensured by extensions to the checksum techniques that are encountered in reliable transport and data link protocols/ "
    # teacher_answer = "Encryption of the message must be done to prevent hacking. Privacy of the senders and receiver parties must be ensured. The integrity of the message must remain and should not be manipulated. Verification of the parties concerned is necessary."
    print("Student Answer:")
    processed_student_answer = text_processing(Regex(student_answer))
    print("Teacher Answer:")
    processed_teacher_answer = text_processing(Regex(teacher_answer))
    matched_phrases,Dict,noun_count,adj_count,verb_count = match_keywords(Regex(student_answer),Regex(teacher_answer))
    sentence = nlp (Regex(student_answer))
    keyword_match_list=list(set([sentence[start:end].text for match_id, start, end in matched_phrases]))
    print(keyword_match_list)
    # matched_contours=[locations[words] for words in keyword_match_list]
    # for c in matched_contours:
    #   x, y, w, h = cv2.boundingRect(c)
    #   print(x+' '+y+' '+w+' '+h+' ')
    marks_eval(Regex(student_answer),keyword_match_list,Dict,noun_count,adj_count,verb_count,marks)
    # tokenised_answer = word_tokenize(Regex(student_answer))
    return keyword_match_list


# nlp_main('H')
# if __name__ == "__main__":
#   main()
    
