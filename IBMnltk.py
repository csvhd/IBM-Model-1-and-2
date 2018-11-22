from nltk.translate import AlignedSent, IBMModel1, IBMModel2 #for the inbuilt IBM Model 1 and 2 functions
import json #for parsing the data file
import time #for calculation of runtime

FILE = 'data_german.json' #input file
SOURCE_LANGUAGE = 'gr' #tag of source language
DESTINATION_LANGUAGE = 'en' #tag of destination language
NUMBER_OF_ITERATIONS = 40 #number of iterations to run the model for

#method for executing the IBM Model 2
def IBM_Model_1(corpus):
    bitext = []
    for x in corpus:
        bitext.append(AlignedSent(x[SOURCE_LANGUAGE].split(), x[DESTINATION_LANGUAGE].split()))
    print("IBM MODEL 1 :")
    print("")
    #calling the inbuilt nltk function for IBM1
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

#method for executing the IBM Model 2
def IBM_Model_2(corpus):
    bitext = []
    for x in corpus:
        bitext.append(AlignedSent(x[SOURCE_LANGUAGE].split(), x[DESTINATION_LANGUAGE].split()))
    print("IBM MODEL 2 :")
    print("")
    ibm2 = IBMModel2(bitext, NUMBER_OF_ITERATIONS)
    #pretty(ibm2.translation_table)
    for test in bitext:
        print("Source sentence:")
        print(test.words)
        print("Destination sentence:")
        print(test.mots)
        print("Alignment:")
        print(test.alignment)
        print("")
    print("----------------------------------------")
    return ibm2.translation_table, bitext

#main function
def main():
    start = time.time()
    #parsing the json file
    with open(FILE, 'r') as f:
        corpus = json.load(f)
    #calling the methods for both the models
    Model1_table, aligned1 = IBM_Model_1(corpus)
    Model2_table, aligned2 = IBM_Model_2(corpus)
    #printing runtime
    print("")
    print("Time:")
    print(time.time() - start)
    
main()