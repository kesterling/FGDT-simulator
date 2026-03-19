
# This file runs the coverage test.The test provides the LLM with a known list of
# human arguments on a topic, and then asks the LLM to evaluate each individual
# response to decide if the reason matches one of the arguments, or none of them.



# import os
import sys # Module that helps with file in
sys.path.append('../')
import GoogleGemini as gemini # Importing different file ("GoogleGemini.py")
# import json # Module that helps with importing and reading files!
import pandas as pd # Used for data descriptions + manipulation
import numpy as np
# rng = np.random.default_rng()



# This function builds a prompt to assign the LLM the classification task, and it returns an integer
def getArgumentMatch(topic: str, arguments: list[str], reason: str) -> int:
    # Create the prompt 
    prompt = f'You are a helpful assistant tasked with classifying argumentative statements on the topic of {topic}. '
    prompt += 'To complete this task, I will first provide you a list of reference arguments, '
    prompt += 'and then I will provide you single candidate argument. You will need to decide which '
    prompt += 'reference argument is closest to the candidate argument. If the candidate argument is '
    prompt += 'is unlike all of the reference arguments, then you will indicate the candidate argument '
    prompt += 'does not have a match.'
    prompt += 'The reference argument list is:\n'
    for i, argument in enumerate(arguments):
        prompt += f'Argument {i+1}: {argument.capitalize()}. '
    prompt += f'The candidate argument is {reason}'
    prompt += 'Please choose the reference argument that is the closest match to the candidate argument.'
    prompt += 'Format your response by only reporting the number of the argument in the reference list. For example, if the closest '
    prompt += 'match is the second argument, then only respond with the number 2. If the candidate argument '
    prompt += 'does not match any of the reference arguments, then only respond with the number 99.'
    # Get the LLM response
    choice = gemini.AskGoogleGemini(prompt)
    choice = int(choice)
    return choice


def main():
    # SET USER PARAMS HERE
    # enter the directory number that stores the results files; this is the first line of the parameter_settings.txt file
    dirnum = 4260266019
#    dirnum = 9870934131 # this is a 6 persona dataset for testing
    # enter the general topic (no more than several words); do not include punctuation at the end
    topic = 'the adoption of Nuclear Power'
    # enter the reference arguments here; do not include punctuation at the end
    arguments = [
        'Nuclear power has zero emissions and can reliably and inexpensively support an electricity grid',
        'Nuclear power is a perfect complement to weather-dependent renewable energies',
        'Nuclear power protects American global national security interests',
        'Nuclear power is dangerous, from the risk of meltdowns to the problem of nuclear waste',
        'Nuclear power is expensive to initiate and delays implementation of truly sustainable and renewable energy sources',
        'Nuclear power is inextricably linked to lethal nuclear weapons',
    ] 
    # the arguments are from https://www.britannica.com/procon/nuclear-power-debate
    # END USER PARAMS

    # read the results file containing the candidate arguments
    directory_name = f'Results/trial_id_{dirnum}'
    tempdf = pd.read_csv(directory_name+'/responsesDF_n0.csv')
    responsesDF = tempdf.to_dict(orient='list')
    # print(responsesDF['reasoning'][2])


    # Initialize LLMs
    gemini.InitGoogleGemini()

    # Initialize random seed
    np.random.seed(5591) # last four digits of Kevin's office number

    print('Matching started.')
    print('=='*40)

    responsesDF['argument']=[]
    i = 0
    for reason in responsesDF['reasoning']:
        # call LLM model which returns an integer
        response = getArgumentMatch(topic, arguments, reason)
        responsesDF['argument'].append(response)
        i += 1
        print(i)

    # # save DF to file
    responsesDF = pd.DataFrame(responsesDF)
    responsesDF.to_csv(directory_name+'/responsesDF_n0_matches.csv', index=False)

    print('=='*40)
    print('Matching completed.')



    # End

if __name__ == '__main__':
    main()


