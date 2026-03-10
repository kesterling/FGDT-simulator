
# this file creates a frequency pipeline, with the goal of matching frequencies with 
# national sample baselines such as PEW. Here's an example of a PEW baseline for nuclear
# power:
# https://www.pewresearch.org/short-reads/2025/10/16/support-for-expanding-nuclear-power-is-up-in-both-parties-since-2020/

# Summearizations and Round2 and Round3 use a RAG-like logic so it's only necessary 
# to get the Round1 frequencies correct so they are matched to a national baseline.

# add to prompt: According to PEW, x% of Republicans etc.

import sys # Module that helps with file in
sys.path.append('../')
import GoogleGemini as gemini # Importing different file ("GoogleGemini.py")
import GroqLlm as llm # Importing different file ("GroqLlm.py")
import digitaltwins_promptblocks as dtb # prompt blocks for digital twin personalities
import json # Module that helps with importing and reading files!
import pandas as pd # Used for data descriptions + manipulation
import numpy as np
rng = np.random.default_rng()

# load the personalities file
tempdf = pd.read_csv('twin_personalities_file.csv')
twinpersonalities = tempdf.to_dict(orient='list')
# print(twinpersonalities['region'][1])

# twin version changes the prompt and add personality fields to the argument
# This function builds a prompt, giving the llm a persona to take on, to send to the llm, the response is recieved as a .JSON
def GetStandpointReasoning(covdict: dict, topic: str, background: str, standpointOptions: list[str], n=0, force=False) -> tuple[int, str]:
# covdict is a dictionary with one record
    "Get the given LLM's standpoint and reasoning on the topic"
    # First, check that there is a prompt block for each element of covdict. 
    # It is fine though to omit covariates, its just you can't include covariates if they don't have a prompt block
    existing_prompt_blocks=['model', 'education', 'race', 'gender', 'politics', 'region', 
                            'pgreen', 'priskaversion']
    for k in covdict.keys():
        if k not in existing_prompt_blocks: 
            print('***Error -- you added a covariate that does not have a prompt block!')
            sys.exit()
    # Create the prompt for this LLM agent
    # You must have a prompt block for each element of cov_list 
    # Each block of prompt text allows for missing data ("None") where necessary
    prompt = 'You will adopt the personality of '
    if 'education' in covdict.keys():
        if covdict['education']=="None":
            pass
        else:
            prompt += f'a {covdict['education']}-educated, '
    if 'race' in covdict.keys():
        if covdict['race']=="None":
            pass
        elif covdict['race']=="Other":
            prompt += 'Non-White, '
        else:
            prompt += f'{covdict['race']}, '
    if 'gender' in covdict.keys():
        if covdict['gender']=="None":
            pass
        else:
            prompt += f'{covdict['gender']}, '
    if 'politics' in covdict.keys():
        if covdict['politics']=="None":
            pass
        else:
            prompt += f'{covdict['politics']}-leaning '
    prompt += 'focus group participant discussing '
    prompt += topic
    # prompt += 'Other aspects of your personality relevant to this discussion are that you: '
    prompt += dtb.digitaltwins_blocks(covdict, prompt)
    prompt += f'You must select from one of the following possible standpoints on {topic} and briefly provide your reasoning for doing so in 1-2 sentences. ' # be sure to correct for the number of options
    prompt += 'The possible standpoints are:\n'
    for i, standpoint in enumerate(standpointOptions):
        prompt += f'Standpoint {i+1}: {standpoint.capitalize()}.'
    prompt += ' '*n # Used for caching results from individual runs
    # caches individual runs because adding a space at the end of the prompt will change it's "numeric value" making considered a new prompt
    prompt += '\n' 
    prompt += background
    prompt += '\n' 
    prompt += 'Format your response in JSON format such as {"Standpoint 2": "your reasoning"}:\n\n'
    # print(prompt)

    # Get the LLM response
    if('gemini' in covdict['model']):
        response = gemini.AskGoogleGemini(prompt, model=covdict['model'], force=force)
    else:
        response = llm.AskGroq(prompt, model=covdict['model'], force=force)
    
# Make sure qwen doesnt output its thinking steps
    if(response.find('<think>') == 0):
        index = response.find('</think>') + len('</think>')
        response = response[index:]
        while(response[0] == '\n'):
            response = response[1:]


    # Format as JSON
    startIndex = response.find('{')
    endIndex = response.find('}')+1
    response = response[startIndex:endIndex]
    try:
        response = json.loads(response)
    except:
        # If JSON format failed, try one more time before quitting
        print('==== RESPONSE: ')
        print(response)
        print('MODEL: ')
        print(covdict['model'])
        print(f'==== ERROR failed to extract JSON format. Trying again. ====')
        # these print statments will return the parameters of the current run to see which model was used and the response recieved!

        # Get the LLM response
        if('gemini' in covdict['model']):
            response = gemini.AskGoogleGemini(prompt, model=covdict['model'], force=True)
        else:
            response = llm.AskGroq(prompt, model=covdict['model'], force=True)

        # Make sure qwen doesnt output its thinking steps
        if(response.find('<think>') == 0):
            index = response.find('</think>') + len('</think>')
            response = response[index:]
            while(response[0] == '\n'):
                response = response[1:]


        # Format as JSON
        startIndex = response.find('{') # finds the brackets ( { } )  to extract the text response
        endIndex = response.find('}')+1
        response = response[startIndex:endIndex]

        try:
            response = json.loads(response)
            print('Resolved.')
            print('-----------------')
        except:
            print('PROMPT:')
            print(prompt)
            print('--')
            print('RESPONSE:')
            print(response)
            print('--')
            print('MODEL:')
            print(covdict['model'])
            print('=='*40)
            print('ERROR: Could not extract JSON')
            quit()
     # same as before; these print statments will return the parameters of the current run to see which model was used and the response recieved!
    
    
    # Extract the standpoint and reasoning
    standpoint = list(response.keys())[0] # Extract standpoint from JSON ex: "Standpoint 1"
    standpoint = int(standpoint.split()[1]) # Ex: "Standpoint 1" -> 1
    reasoning = list(response.values())[0] # Extract reasoning from JSON

    # Strip surrounding quotes from response if they exist
    if(reasoning[0] == '"' and reasoning[-1] == '"'):
        reasoning = reasoning[1:-1]

    return standpoint, reasoning

# twin version changes the sampling and removes genderOptions, politicsOptions, and educationOptions from the arguments
# and omits prytaneumExperiment sampling block
def RunExperiment(models: list[str], cov_list: list[str], topic: str, background: str, standpointOptions: list[str], # genderOptions: list[str], politicsOptions: list[str], educationOptions: list[str], 
                  sampleSize=8, n=0, saveToFile=False, force=False):
    """Returns a rectangular dataset of paramter settings, standpoints chosen and reasoning for each twin
    @Param n represents the current run number, and is used to cache results from each individual run."""

    # Round 1: Get standpoints and reasonings

    # This is the old pipeline that randomizes personalities -- no longer used
    # # Get all possible LLM personalities
    # personalities = [] # [(model, education, gender, politics), ...]
    # for model in models:
    #     for education in educationOptions:
    #         for gender in genderOptions:
    #             for politics in politicsOptions:
    #                 personalities.append((model, education, gender, politics))
    #                 # saves all personality combinations to the list "personalities"

    # # Choose a random sample of them for this experiment run
    # sampleIndexes = np.random.choice(len(personalities), sampleSize) # Get their indexes
    # samplePersonalities = [] # [(model, education, gender, politics), ...]
    # for index in sampleIndexes: # Fetch the personalities from this list of indexes
    #     samplePersonalities.append(personalities[index])

    covdict_list=[] # is a list of dictionaries where each row gets sent to GetStandpointReasoning
    responsesRound1list=[] # initialize this list to store results dataframe

    # Draw sample of twins with N=sampleSize
    # Note: replace=False so do not set sampleSize > 2058 (!) otherwise you will get an error
    sampleIndexes = np.random.choice(len(twinpersonalities['region']), sampleSize, replace=False) # Get their indexes

    for i in sampleIndexes:
        x = {'model': models[np.random.randint(0, 4)]}
    # Be sure to specify correct cov_list in user params with list of covariates 
    # (but not including models, standpointNum or reasoning)
        for j in cov_list:
            x[j]=twinpersonalities[j][i]
        covdict_list.append(x)

    # Run the experiment using this sample by sending covdict: dict to GetStandpointReasoning, 
    # covdict is a single element of covdict_list
    for covdict in covdict_list:
        # Get the standpoint and reasoning for this LLM
        standpointNum, reasoning = GetStandpointReasoning(covdict, topic, background, standpointOptions, n, force)

# append standpointNum and reasoning (in that order) to 
        # Save the response as a new row
        covdict['standpointNum']=standpointNum
        covdict['reasoning']=reasoning
        responsesRound1list.append(covdict)

    responsesRound1DF = pd.DataFrame(responsesRound1list)

    # Save Round 1 results to file
    if(saveToFile):
        y=int(rng.integers(10**10))
        responsesRound1DF.to_csv(f'Results/responsesRound1_n{y}.csv', index=False)
    # outputs file after round 1

    # returns None


# twins pipeline removes politicsOptions, genderOptions and educationOptions from user params and arg list for RunExperiment
def main():
    "Simulate a focus group using LLM agents"
    # SET USER PARAMS HERE
    models = [
        'gemini-2.5-flash',  
        'llama-3.3-70b-versatile',
        'qwen/qwen3-32b',
        #'deepseek-r1-distill-llama-70b',
        #'gemma2-9b-it',
        'openai/gpt-oss-120b',
    ]
    # the entries of cov_list must be fields in twinpersonalities
    cov_list=['gender', 'education', 'politics', 'pgreen']
    topic = 'the adoption of Nuclear Power. '
    standpointOptions = [
        'implement rapid expansion of and investment in nuclear power ',
        'maintain the current nuclear energy operations without change ',
        'phase out existing nuclear plants and halt new construction ',
#        'prioritize researching improved implementations of nuclear power. ',
    ] 
    background = '''To help you choose an option, please know that according to PEW Research Center, 
    59 percent of US residents favor expanding nuclear power. 73 percent of men 
    favor expanding it while only 44 percent of women favor expanding it. 
    69 percent of Republicans favor expanding compared to 52 percent of Democrats. 
    Enviornmentalists have mixed opinions, where some favor expanding it because 
    nuclear has a low-carbon footprint, while others are concerned about the risks of nuclear waste disposal.'''
    sampleSize = 5 # This is the sample size of LLM agents; set this for all experiments
    # numAnswers = 1  # Number of responses per LLM personality - ignored/not used.
    numRuns = 1 # Number of runs per experiment; usually just one for this file
    saveRawToFile = True # Save the raw results of each experiment run to file; keep as True unless testing
    force = False  # set to True if you don't want to use cached answers
    # END USER PARAMS

    # Initialize LLMs
    gemini.InitGoogleGemini()
    llm.InitGroq()

    # Initialize random seed
    np.random.seed(5591) # last four digits of Kevin's office number

    for n in range(numRuns):
        # Output debugging info
        if(n % 1 == 0):
            print('=='*40)
            print(f'Starting run #{n+1} out of {numRuns}...')

    # Start the experiment for the given number of runs
        RunExperiment(models, cov_list, topic, background, standpointOptions,  
                        sampleSize, n, saveRawToFile, force)
        
    print('=='*40)
    print('Simulation completed.')

    # return None
    # End of experiment

if __name__ == '__main__':
    main()


