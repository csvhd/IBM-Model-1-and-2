from nltk.translate import AlignedSent, IBMModel1, IBMModel2 #for the inbuilt IBM Model 1 and 2 functions
from nltk.translate.phrase_based import phrase_extraction #for the inbuilt phrase extractor in nltk
import json #for parsing the data file
from collections import defaultdict #for the custom dictionaries
import time  #for calculation of runtime

FILE = 'data_german.json' #input file
SOURCE_LANGUAGE = 'gr' #tag of source language 
DESTINATION_LANGUAGE = 'en' #tag of destination language
NUMBER_OF_ITERATIONS = 40 #number of iterations to run the model for

#calling the inbuilt IBM1 method
def IBM_Model_1(corpus):
    bitext = []
    for x in corpus:
        bitext.append(AlignedSent(x[SOURCE_LANGUAGE].split(), x[DESTINATION_LANGUAGE].split()))
    print("IBM MODEL 1 :")
    print("")
    #calling the inbuilt IBM Model 1 function
    ibm1 = IBMModel1(bitext, NUMBER_OF_ITERATIONS)
    for test in bitext:
        print("Source sentence:")
        print(test.words)
        print("Destination sentence:")
        print(test.mots)
        print("Alignment:")
        print(test.alignment)
        print("")
    print("----------------------------------------")
    return ibm1.translation_table, bitext

def main():
    start = time.time()
    #parsing the json file
    with open(FILE, 'r') as f:
        corpus = json.load(f)
    Model1_table, aligned1 = IBM_Model_1(corpus)

    alignments_of_1 = []
    words_of_1 = []
    mots_of_1 = []
    #storing the information got from the IBM Model 1 in lists
    for test in aligned1:
        alignments_of_1.append(test.alignment)
        words_of_1.append(test.words)
        mots_of_1.append(test.mots)
    print("")
    c = 0
    #traversing the corpus
    for x in corpus:
        srctext = x[SOURCE_LANGUAGE]
        destext = x[DESTINATION_LANGUAGE]
        align = alignments_of_1[c]
        print("Source sentence:")
        print(words_of_1[c])
        print("Destination sentence:")
        print(mots_of_1[c])
        print("Alignment:")
        print(align)
        print("")
        c = c + 1
        sorted_phrase_score = list()
        #calling the inbuilt function to extract phrases
        phrases = phrase_extraction(srctext, destext, align)
        for i in phrases:
            SOURCE_phrase = i[2]
            DESTINATION_phrase = i[3]
            count_numerator = 0.0
            count_denominator = 0.0
            for y in corpus:
                #checking if both the phrases are in the sentence
                if SOURCE_phrase in y[SOURCE_LANGUAGE] and DESTINATION_phrase in y[DESTINATION_LANGUAGE]:
                    count_numerator = count_numerator + 1
                #checking if the source phrase is in the source sentence
                if SOURCE_phrase in y[SOURCE_LANGUAGE]:
                    count_denominator = count_denominator + 1
            #calculating the phrase score
            phrase_score = count_numerator / count_denominator
            #adding the phrase score to a list
            sorted_phrase_score.append((phrase_score, SOURCE_phrase, DESTINATION_phrase))
        #printing the output in descending order of the phrase score
        for values in sorted(sorted_phrase_score, reverse=True):
            print("Source phrase:")
            print(values[1])
            print("Destination phrase:")
            print(values[2])
            print("Phrase Score:")
            print(values[0])
            print("")

    #printing runtime
    print("")
    print("Time:")
    print(time.time() - start)
main()