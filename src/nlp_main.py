# !python -m spacy download en_core_web_lg
import spacy
from spacy.lang.en import English
import en_core_web_lg
nlp = en_core_web_lg.load()
# nlp = spacy.load('en_core_web_lg')
from spacy.matcher import PhraseMatcher
phrase_matcher = PhraseMatcher(nlp.vocab)
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
# pip install -U sentence-transformers
import regex as re
import scipy
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('bert-base-nli-mean-tokens')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
from scipy import spatial

#function to extract keywords
def extract_nouns_adj(lines):
  tokenized = nltk.word_tokenize(lines)
  list_word_tags = nltk.pos_tag(tokenized)
  # function to test if something is a noun or adj or a verb
  is_noun_adj = lambda pos: pos[:2] == 'NN' or pos[:2] == 'JJ' or pos[:2] == 'VB'
  # do the nlp stuff
  # tokenized = nltk.word_tokenize(lines)
  nouns_adjs = [(word,pos) for (word, pos) in list_word_tags if is_noun_adj(pos)] 
  print(nouns_adjs)
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

# function that computes semantic similarities between the answers
def semantic_similarity(corpus,queries):
  corpus_embeddings = model.encode(corpus)
  query_embeddings = model.encode(queries)
  closest_n = 3
  for query, query_embedding in zip(queries, query_embeddings):
      distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0] 
      #find the cosine similarity 
      # print("\n\n======================\n\n")
      results = zip(range(len(distances)), distances)
      # print(results)
      results = sorted(results, key=lambda x: x[1])
      # print("\n\n======================\n\n")
      # print(results)
      print("\n\n======================\n\n")
      print("Query:", query)
      print("\nTop 3 most similar sentences in corpus:")

      for idx, distance in results[0:closest_n]:
          print(corpus[idx].strip(), "(Score: %.4f)" % (1-distance))

# method to find cosine similarity
def cosineSimilarity(vect1, vect2):
    # return cosine distance
    return 1 - spatial.distance.cosine(vect1, vect2)

def createKeywordsVectors(keyword, nlp):
    doc = nlp(keyword)  # convert to document object

    return doc.vector

# method to find similar words
def getSimilarWords(keyword_and_pos, nlp):
    similarity_list = []
    keyword,og_tag = keyword_and_pos 
    # print(keyword_and_pos[0])
    keyword_vector = createKeywordsVectors(keyword, nlp)
    for tokens in nlp.vocab:
        if (tokens.has_vector):
            if (tokens.is_lower):
                if (tokens.is_alpha):
                    # print(tokens.text)
                    similarity_list.append((tokens, cosineSimilarity(keyword_vector, tokens.vector)))

    similarity_list = sorted(similarity_list, key=lambda item: -item[1])
    similarity_list = similarity_list[:30]
    # print('3
    top_similar_words = [item[0].text for item in similarity_list]
    # og_tag = nltk.pos_tag([keyword])[0][1]
    # print(og_tag)
    is_noun_adj = lambda pos: pos[:2] == og_tag
    top_similar_words = [word for (word,tag) in [nltk.pos_tag([word])[0] for word in top_similar_words] if is_noun_adj(tag)]
    
    top_similar_words = top_similar_words[:3]


    top_similar_words.append(keyword)
    # print('4')
    for token in nlp(keyword):
        top_similar_words.insert(0, token.lemma_)

    top_similar_words = list(set(top_similar_words))
    # print('5')
    # top_similar_words = [words for words in top_similar_words if enchant_dict.check(words) == True]

    return top_similar_words

def keyword_count(test_list):
  noun_count = 0
  adj_count = 0
  verb_count = 0
  for (text,tag) in test_list:
    tag=tag[:2]
    print(str(text)+': '+tag)
    if tag == 'NN':
      noun_count+=1
    elif tag == 'JJ':
      adj_count+=1
    else :
      verb_count+=1
  return noun_count,adj_count,verb_count

def match_keywords(student_answer,teacher_answer):
  lemmatizer = WordNetLemmatizer()
  new_teacher_answer = ' '.join([lemmatizer.lemmatize(t) for t in word_tokenize(Regex(teacher_answer)) if t not in stop_words])
  test_list = []
  test_list = list(set(extract_nouns_adj(new_teacher_answer)))
  noun_count,adj_count,verb_count= keyword_count(test_list)
  print("Keywords found in teacher's answer:")
  print(test_list)
  new_test_list = []
  print("Similar word generation for the keywords:")
  for words in test_list:
    print(words)
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
  # noun_count,adj_count,verb_count= keyword_count(final_test_list)
  print(str(noun_count)+" "+str(verb_count)+" "+str(adj_count))
  phrase_matcher.add('FinalPhraseMatcher', None, *patterns)
  sentence = nlp (student_answer)

  matched_phrases = phrase_matcher(sentence)
  for match_id, start, end in matched_phrases:
    string_id = nlp.vocab.strings[match_id]  
    span = sentence[start:end]                   
    print(str(span.text)+" : word location in student's answer: "+str(start))

  return matched_phrases

def nlp_main(teacher_answer:str, student_answer:str):
# def main():
    # student_answer = "Confidentiality: Only the sender and the receiver should be able to understand the contents of the transmitted message. The message may be encrypted due to hackers. This is the most commonly perceived meaning of secure communication. Authentication: Both sender and receiver should be able to confirm the identity of the other party involved in the communication i.e to cofirm that the other party is indeed who or what they claim to be. Message integrity and non-repudiation: Even if the sender and receiver are authenticated, they ensure that the context of their communication is not altered. Message integrity can be ensured by extensions to the checksum techniques that are encountered in reliable transport and data link protocols/ "
    # teacher_answer = 'Encryption of the message must be done to prevent hacking. Privacy of the senders and receiver parties must be ensured. The integrity of the message must remain and should not be manipulated. Verification of the parties concerned is necessary.'
    processed_student_answer = text_processing(student_answer)
    processed_teacher_answer = text_processing(teacher_answer)
    print(getSimilarWords(('privacy','NN'), nlp))
    semantic_similarity(processed_student_answer,processed_teacher_answer)
    matched_phrases=match_keywords(Regex(student_answer),Regex(teacher_answer))
    tokenised_answer = word_tokenize(Regex(student_answer))

# nlp_main('H')
    
