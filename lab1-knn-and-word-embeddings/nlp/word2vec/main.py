# import modules & set up logging
import nltk
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading WordNet data...")
    nltk.download('wordnet')
    nltk.download('omw-1.4')
import gensim, logging, numpy as np
import help_functions as hf
import time

#@author: The first version of this code is the courtesy of Vadim Selyanik

dimensions = [10, 100, 1000, 2000, 4000] # possible dimensions for word embeddings
num_runs = 10
results = {}

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
lemmatizer = nltk.WordNetLemmatizer() # create a lemmatizer

sentences = []
file = open("lemmatized.text", "r")

for line in file: # read the file and create list which contains all sentences found in the text
    sentences.append(line.split())
file.close()
# train word2vec on the two sentences

print("Testing training on dimensions: " + str(dimensions))
print("Number of runs for each dimension: " + str(num_runs))

start_time = time.time()

for dimension in dimensions:
    results[dimension] = []
    
    for run in range(num_runs):
        print("Dimension: " + str(dimension) + "; Run: " + str(run+1)+ "/" + str(num_runs))
        threshold = 0.00055 # parameter for Word2vec

        # 
        model = gensim.models.Word2Vec(sentences, min_count=1, sample=threshold, sg=1,vector_size=dimension) # create model using Word2Ve with the given parameters
        words = list(model.wv.key_to_index.keys())
        #
        print(len(words)) # check the length of the vocabulary which was formed by Word2Vec

        #The rest implements passing TOEFL tests
        i = 0 #counter for TOEFL tests
        number_of_tests = 80
        text_file = open('new_toefl.txt', 'r')
        right_answers = 0 # variable for correct answers
        number_skipped_tests = 0 # some tests could be skipped if there are no corresponding words in the vocabulary extracted from the training corpus

        while i < number_of_tests:
                    line = text_file.readline() #read line in the file
                    words = line.split() # extract words from the line
                    try:
                        words = [lemmatizer.lemmatize(lemmatizer.lemmatize(lemmatizer.lemmatize(word, 'v'), 'n'), 'a') for word in
                                words] # lemmatize words in the current test
                        vectors = []
                        if words[0] in words: # check if there embedding for the query word
                            k = 1 #counter for loop iterating over 5 words in the test
                            vectors.append(model.wv[words[0]])
                            while k < 5:
                                if words[k] in words: # if alternative has the embedding
                                    vectors.append(model.wv[words[k]]) #assing the learned vector
                                else: 
                                    vectors.append(np.random.randn(dimension)) #assing random vector
                                k += 1
                            right_answers += hf.get_answer_mod(vectors) #find the closest vector and check if it is the correct answer

                    except KeyError: # if there is no representation for the query vector than skip
                        number_skipped_tests += 1
                        print("skipped test: " + str(i) + "; Line: " + str(words))
                    except IndexError:
                        print(i)
                        print(line)
                        print(words)
                        break
                    i += 1
        text_file.close()
        
        accuracy = 100 * float(right_answers) / float(number_of_tests) #get the percentage
        print("Threshold ferq = "+ str(threshold))
        print("Percentage of correct answers: " + str(accuracy) + "%")
        results[dimension].append(accuracy)
        print(f"Run {run+1} completed. Accuracy: {accuracy:.2f}%")
        print("Skipped tests: " + str(number_skipped_tests))

end_time = time.time()
total_time = end_time - start_time
#%%
print("Final results:\n")

results_filename = "results_gensim_word2vec.txt"
hf.save_results_to_file(results_filename, results, total_time)