import json #for extracting data from the .json files
from collections import defaultdict #for making higher dimensional dictionaries
import time

SOURCE_LANGUAGE = 'gr' # the source language tag in the json file
DESTINATION_LANGUAGE = 'en' # the destination language tag in the json file
FILE = 'data_german.json' #data file with the translations
NUMBER_OF_ITERATIONS = 40 #number of iterations to run the EM algorithm

#method to get all the words in the corpus in a more accessible dictionary where dictionary maps the tag of the language to the  words
def get_words(corpus):
    corpus_words = defaultdict(set)
    for x in corpus:
        for y in x:
            words = x[y].split()
            for word in words:
                corpus_words[y].add(word)
    return corpus_words

#method to initialize the transition_probabilities to 1/(length of the sentence in the source language)
def initialize_uniform(corpus):
    uni_dict = defaultdict(dict)
    words = get_words(corpus)
    for word_SOURCE in words[SOURCE_LANGUAGE]:
        for word_DESTINATION in words[DESTINATION_LANGUAGE]:
            uni_dict[word_SOURCE][word_DESTINATION] = 1/len(words[SOURCE_LANGUAGE])
    return uni_dict

#method to train the EM model the specified number of iterations
def EM(corpus):
    
    words = get_words(corpus)
    s_total = {}
    #initializing required dictionaries
    for word in words[SOURCE_LANGUAGE]:
        s_total[word] = 0.0
    
    previous = initialize_uniform(corpus) 
    iterations = 0

    while iterations < NUMBER_OF_ITERATIONS:
        current = previous
        count = defaultdict(dict)
        total = defaultdict(dict)

        #initializing count
        for word_SOURCE in words[SOURCE_LANGUAGE]:
            for word_DESTINATION in words[DESTINATION_LANGUAGE]:
                count[word_SOURCE][word_DESTINATION] = 0

        #initializing total
        for word in words[DESTINATION_LANGUAGE]:
            total[word] = 0

        #computing normalization
        for (es, fs) in [(pair[SOURCE_LANGUAGE].split(), pair[DESTINATION_LANGUAGE].split()) for pair in corpus]:
            for e in es:
                s_total[e] = 0
                for f in fs:
                    s_total[e] += current[e][f]

        #collecting count
            for e in es:
                for f in fs:
                    count[e][f] += (current[e][f] / s_total[e])
                    total[f] += current[e][f] / s_total[e]

        #estimating the transition probabilities in the table
        for f in words[DESTINATION_LANGUAGE]:
            for e in words[SOURCE_LANGUAGE]:
                current[e][f] = count[e][f] / total[f]

        previous = current
        iterations += 1

    return current

#method to return the translations with the maximum translation probabilities in form of a dictionary
def result_with_maxprob(current):
    final_result = {}
    SOURCE_keys = current.keys()
    DESTINATION_words = list(current.values())
    DESTINATION_keys = DESTINATION_words[0].keys()

    for word_SOURCE in SOURCE_keys:
        max_prob = 0.0
        max_DESTINATION_word = ""
        for word_DESTINATION in DESTINATION_keys:
            if current[word_SOURCE][word_DESTINATION] >= max_prob:
                max_prob = current[word_SOURCE][word_DESTINATION]
                max_DESTINATION_word = word_DESTINATION
        final_result[word_SOURCE] = max_DESTINATION_word

    return final_result

#method to print the alignment as per the IBM nltk module standard
def print_result(result, corpus):
    print("")
    for x in corpus:
        i = 0
        print("Source sentence:")
        print(x[SOURCE_LANGUAGE].split())
        print("Destination sentence:")
        print(x[DESTINATION_LANGUAGE].split())
        print("Alignment:")
        for word_SOURCE in x[SOURCE_LANGUAGE].split():
            j = 0
            for word_DESTINATION in x[DESTINATION_LANGUAGE].split():
                if word_DESTINATION == result[word_SOURCE]:
                    align = str(i)+"-"+str(j)+" "
                    print(align, end= "")
                    j = j + 1
                    break
                j = j + 1
            i = i + 1
        print("")
        print("")
          
#main method to start the program
def main():
    start = time.time()
    #parsing the json file and storing it in an object
    with open(FILE, 'r') as f:
        corpus = json.load(f)
    
    #calling the EM iteration method
    probabilities = EM(corpus)
    #calculating the maximum probable translation
    result_table = result_with_maxprob(probabilities)
    #printing the result as per the IBM nltk module
    print_result(result_table, corpus)
    #printing the runtime
    print("Time:")
    print(time.time() - start)

main()
