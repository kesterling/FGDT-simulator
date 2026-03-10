
# This file runs the focus group simulations using the structure of three rounds:
# 1) choice and reasoning, 2) counterargument, 3) response to a counterargument

# First use Round1_frequencies.py to try to match option response frequencies with 
# national sample baselines such as PEW. Here's an example of a PEW baseline for nuclear
# power:
# https://www.pewresearch.org/short-reads/2025/10/16/support-for-expanding-nuclear-power-is-up-in-both-parties-since-2020/
# Summarizations and Round2 and Round3 use a RAG-like logic so it's only necessary 
# to get the Round1 frequencies correct so they are matched to a national baseline.


import os
import sys # Module that helps with file in
sys.path.append('../')
import GoogleGemini as gemini # Importing different file ("GoogleGemini.py")
import GroqLlm as llm # Importing different file ("GroqLlm.py")
import digitaltwins_promptblocks as dtb # prompt blocks for digital twin personalities
import json # Module that helps with importing and reading files!
import pandas as pd # Used for data descriptions + manipulation
import numpy as np
from sentence_transformers import SentenceTransformer # Assigns numerical values to text to help with paraphrasing/finding meaning within a sentence (text -> vector)
from sklearn.metrics.pairwise import cosine_similarity # Uses the numerical values assigned to compare them  and guage if sentences have the same meaning!!
rng = np.random.default_rng()

# load the personalities file as a list of dictionaries
tempdf = pd.read_csv('twin_personalities_file.csv')
twinpersonalities = tempdf.to_dict(orient='list')
# print(twinpersonalities['region'][1])

# twin version changes the prompt and add personality fields to the argument
# This function builds a prompt, giving the llm a persona to take on, to send to the llm, the response is recieved as a .JSON
def GetStandpointReasoning(covdict: dict, topic: str, background: str, standpointOptions: list[str], n=0, force=False) -> tuple[int, str]:
# **** covdict is a dictionary with one record, containing 'model' plus covariates
    "Get the given LLM's initial standpoint and reasoning on the topic"
    # First, check that there is a prompt block for each element of covdict. 
    existing_prompt_blocks=dtb.existing_prompt_blocks()
    for k in covdict.keys():
        if k not in existing_prompt_blocks: 
            print('***Error -- you added a covariate that does not have a prompt block!')
            sys.exit()
    # You must have a prompt block for each element of cov_keys 
    # It is fine though to omit covariates, its just you can't include covariates if they don't have a prompt block
    # Create the prompt for this LLM agent
    # Each block of prompt text allows for missing data ("None") where necessary
    # dtb is the prompt blocks for digitaltwins
    prompt = dtb.digitaltwins_blocks(covdict, topic)
    prompt += f'You must select from one of the following possible standpoints on {topic} and briefly provide your reasoning for doing so in 1-2 sentences. ' # be sure to correct for the number of options
    prompt += 'The possible standpoints are:\n'
    for i, standpoint in enumerate(standpointOptions):
        prompt += f'Standpoint {i+1}: {standpoint.capitalize()}. '
    prompt += ' '*n # Used for caching results from individual runs
    # caches individual runs because adding a space at the end of the prompt will change it's "numeric value" making considered a new prompt
    prompt += '\n' 
    prompt += background
    prompt += '\n' 
    if ('qwen' in covdict['model']):
        prompt+='/nothink'  # this saves inference costs and reduces JSON load errors
    prompt += 'Format your response in JSON format such as {"Standpoint 2": "your reasoning"}:\n\n'
    # print(prompt)

    # Get the LLM response
    if('gemini' in covdict['model']):
        response = gemini.AskGoogleGemini(prompt, model=covdict['model'], force=force)
    else:
        response = llm.AskGroq(prompt, model=covdict['model'], force=force)
    
    # Remove qwen thinking field
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
        print('==== ERROR failed to extract JSON format. Trying again. ====')
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
            sys.quit()
     # same as before; these print statments will return the parameters of the current run to see which model was used and the response recieved!
    
    
    # Extract the standpoint and reasoning
    standpoint = list(response.keys())[0] # Extract standpoint from JSON ex: "Standpoint 1"
# add next lines to repository!
    standpoint = standpoint.split()[1]  # Ex: "Standpoint 1" -> 1
    if standpoint[-1]==':':
        standpoint=standpoint[:-1]
    standpoint = int(standpoint) 
    reasoning = list(response.values())[0] # Extract reasoning from JSON

    # Strip surrounding quotes from response if they exist
    if(reasoning[0] == '"' and reasoning[-1] == '"'):
        reasoning = reasoning[1:-1]

    return standpoint, reasoning

# This function takes in the result of the GetStandpointReasoning function as its parameter and returns the responses as a list
def SummarizeResponses(topic: str, standpoint: list|str, responses: list, counterargument=False, n=0, force=False) -> list[str]:
    """Summarize all given responses that justify the given standpoint.
    Set counterargument to True if the responses are instead arguing against the given standpoint.
    Set standpoint as a list of standpoints to not summarize in terms of any particular standpoint."""
    if(len(responses) < 1):
        return [] # if no responses are captured (from function before) then only an empty lisy will be returned!

    prompt = f'You are a helpful assistant tasked with summarizing argumentative statements on the topic of {topic}. '
    prompt += 'I have gathered a list of statements that participants in an online focus group platform have provided as their '
    # gives the LLM a command, which in this case is to summarize the arguments and reasoning.

    # If a list of standpoints are given, we are in round 3
    if(isinstance(standpoint, list)): # "isinstance" checks if the variable "standpoint" is a list
        standpoints = ', '.join(standpoint) # turns the list into a single string

        prompt += f'reasoning for supporting one of the following standpoints: {standpoints}. '
        prompt += 'Some participants may have voiced counterarguments against other viewpoints, while others may have only stated their own views. '
    # Otherwise, we are in round 1 or 2
    else:
        # if there is not enough responses to join into a string, it will just be sinularly pushed through here
        if(not counterargument): # if the LLM gave a response AGREEING to the standpoint:
            prompt += f'reasoning to {standpoint}. '
        else: # if the LLM gave a response DISAGREEING to the standpoint:
            prompt += f'reasoning to not {standpoint}. '
    

    # building the prompt to have LLM summarize views
    if(len(responses) == 1): # if there is only 1 recorded response, return 1 summary
        prompt += 'Summarize the following statements as exactly one concise single summary. '
        prompt += 'Your summary should be worded as passive tone viewpoints, so that it captures '
    else: # if there is more than 1 response, return MAXIMUM 5 unique summarizations 
        prompt += 'Summarize the following statements as no more than five concise summaries. '
        prompt += 'Your summaries should be worded as passive tone viewpoints, so that they capture '

    prompt += 'as many viewpoints presented in the given statements as possible. '

    if(len(responses) == 1):
        prompt += 'Start the summary on a new line with a dash. '
    else:
        prompt += 'Use as few summaries as are necessary to achieve this. '
        prompt += 'Start each summary on a new line with a dash. '
    prompt += '\n\n'

    prompt += 'The statements are:\n'
    for i, post in enumerate(responses):
        prompt += f'Statement {i+1}: "{post}"\n'
    prompt += '\n'
        
    prompt += 'Your summarized viewpoints:\n'
    prompt += ' '*n # Used for caching results from individual runs

    response = gemini.AskGoogleGemini(prompt, force=force) # calls LLM
    # reformats to string
    response = response.replace('\n', '') 
    response = response.split('- ')
    response = response[1:]
    return response # list of response summaries 




# LLM will give a coutner argument based on a specific assigned persona 
def GetCounterArgument(covdict: dict, topic: str, standpoint: str, reasoning: str, opposingViewpoint: str, n=0, force=False) -> str:
    "Make the LLM come up with a counterargument for the given reasoning on Nuclear Power"
    # covdict is a dictionary with 'model' and covariates 
    # Create the prompt for this LLM agent
    # This prompt sets up education level, personality, etc, and then requests a counter argument to the current general response 
    # Each block of prompt text allows for missing data ("None") where necessary
    prompt = dtb.digitaltwins_blocks(covdict, topic)
    prompt += f'You have previously stated that your standpoint is to {standpoint}, and your reasoning for this was: "{reasoning}".'
    prompt += '\n\n'
    prompt += f'As part of the focus group discussion, a different viewpoint has been brought up: "{opposingViewpoint}".'
    prompt += '\n'
    prompt += ' '*n # Used for caching results from individual runs
    prompt += '\n' + ' '*0 # 2nd level caching -- ignored in this version
    prompt += f'You must now provide a counterargument for this different viewpoint on {topic} in 1-2 sentences:\n'
    if ('qwen' in covdict['model']):
        prompt+='/nothink'  # this saves inference costs and reduces JSON load errors

    # Get the LLM response
    # The prompt built above is sent to the AI
    if('gemini' in covdict['model']): 
        response = gemini.AskGoogleGemini(prompt, model=covdict['model'], force=force)
    else: 
        response = llm.AskGroq(prompt, model=covdict['model'], force=force)
    
    # Try again if response is empty
    if(len(response) < 1):
        print("Fail1! Try again...")
        print(covdict['model'])
        print('----')
        print(prompt)
        print('-----')
        print(response)
        if('gemini' in covdict['model']): 
            response = gemini.AskGoogleGemini(prompt, model=covdict['model'], force=force)
        else: 
            response = llm.AskGroq(prompt, model=covdict['model'], force=force)
    # Try again one more time if response is empty
    if(len(response) < 1):
        print("Fail2! try one more time...")
        print(covdict['model'])
        print('----')
        print(prompt)
        print('-----')
        print(response)
        if('gemini' in covdict['model']): 
            response = gemini.AskGoogleGemini(prompt, model=covdict['model'], force=force)
        else: 
            response = llm.AskGroq(prompt, model=covdict['model'], force=force)
    if(len(response) < 1):
        print("Fail3! Last attempt tried -- oh well -- force quit")
        print(covdict['model'])
        sys.quit()
        

    # Make sure qwen doesnt output its thinking steps
    if(response.find('<think>') == 0):
        index = response.find('</think>') + len('</think>')
        response = response[index:]
        while(response[0] == '\n'):
            response = response[1:]

    # Strip surrounding quotes from response if they exist
    if(response[0] == '"' and response[-1] == '"'):
        response = response[1:-1]
    
    return response


def GetResponseToCounterArgument(covdict: dict, topic:str, standpoint: str, reasoning: str, counterargument: str, n=0, force=False) -> str:
    "Make the LLM respond to a counterargument for its previously given reasoning on Nuclear Power"
    # covdict is a dictionary with 'model' and covariates 
    # Create the prompt for this LLM agent
    # This prompt sets up education level, personality, etc, and then requests a response to a counter argument to the initial response 
    # Each block of prompt text allows for missing data ("None") where necessary
    prompt = dtb.digitaltwins_blocks(covdict, topic)
    prompt += f'You have previously stated that your standpoint is to {standpoint}, and your reasoning for this was: "{reasoning}".'
    prompt += '\n\n'
    prompt += 'As part of the focus group discussion, a counterargument to your standpoint has been brought up by a participant who '
    prompt += f'said that: "{counterargument}". '
    prompt += '\n\n'
    prompt += 'You must now respond to this counterargument in 1-2 sentences.\n'
    if ('qwen' in covdict['model']):
        prompt+='/nothink'  # this saves inference costs and reduces JSON load errors
    prompt += ' '*n # Used for caching results from individual runs
    prompt += '\n' + ' '*0 # 2nd level caching - ignored in this version

    # Get the LLM response
    # checks for Gemini, if not then send prompt request to Groq
    if('gemini' in covdict['model']):
        response = gemini.AskGoogleGemini(prompt, model=covdict['model'], force=force)
    else:
        response = llm.AskGroq(prompt, model=covdict['model'], force=force)
    
    # Make sure qwen doesnt output its thinking steps
    if(response.find('<think>') == 0):
        index = response.find('</think>') + len('</think>') # strips the <think> output section
        response = response[index:]
        while(response[0] == '\n'):
            response = response[1:]
    
    return response

# this next block runs the experiment for a single run
def RunExperiment(models: list[str], cov_keys: list[str], topic: str, background: str, standpointOptions: list[str], directory_name: str, # genderOptions: list[str], politicsOptions: list[str], educationOptions: list[str], 
                  sampleSize=8, n=0, saveToFile=False, force=False):
    """Returns a rectangular dataset of paramter settings, standpoints chosen, reasoning, counter argument and response to counterargument for each twin
    @Param n represents the current run number, and is used to cache results from each individual run."""
    # models, cov_keys, topic, background, and standpointOptions are user parameters set in main()

    # Round 1: Get standpoints and reasonings
    print('--'*40)
    print('Round 1 underway! ...')
    
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

    responsesList=[] # initialize this list to store personality covariates and results data
    allSummaries=[]
    
    # Draw sample of twins with N=sampleSize
    # Note: replace=False so do not set sampleSize > 2058 (!) otherwise you will get an error
    sampleIndexes = np.random.choice(len(twinpersonalities['gender']), sampleSize, replace=False) # Get their indexes

    for i in sampleIndexes:
        x = {'model': models[np.random.randint(0, 4)]}
    # Be sure to specify correct cov_keys in user params with 'model' plus list of covariates 
    # (including models but not including standpointNum or reasoning)
        for j in cov_keys[1:]:
            x[j]=twinpersonalities[j][i]
        responsesList.append(x)

    # Run the experiment using this sample by sending covdict: dict to GetStandpointReasoning, 
    # covdict is a subset of a responsesList row containing only 'model' and the covariates
    for i, row in enumerate(responsesList):
        # Output some debugging info (prints every 10 responses)
        if((i+1) % 10 == 0 or i == 0 or (i+1) == len(responsesList)):
            print(f'Round 1: Getting Standpoint {i+1} out of {len(responsesList)} (run #{n+1})...')

        covdict = {key: row[key] for key in cov_keys if key in row} # this line is redundant for this round but needed in rounds 2 and 3
        # Get the standpoint and reasoning for this LLM
        standpointNum, reasoning = GetStandpointReasoning(covdict, topic, background, standpointOptions, n, force)
    # append standpointNum and reasoning (in that order) to responsesList
        responsesList[i]['standpointNum']=standpointNum
        responsesList[i]['reasoning']=reasoning

    # Post round 1: Summarize the responses
    # Gather the reasoning reasonings for each standpoint
    responsesDF = pd.DataFrame(responsesList)
    standpointSummaries = {} # {standpointIndex: [summaries], ...}
    for i, standpoint in enumerate(standpointOptions):
        # Get all reasonings using Pandas query and convert to list
        reasonings = responsesDF.loc[responsesDF['standpointNum'] == i+1, 'reasoning']
        reasonings = list(reasonings.values)
        # Get and save the response summaries
        summaries = SummarizeResponses(topic, standpoint, reasonings, False, n, force)
        # summaries is a list of summarized reasonings
        standpointSummaries[i] = summaries
    allSummaries.append(standpointSummaries)
        # standpointSummaries is a dictionary of the round1 summaries lists
        
    # Round 2: Get counterarguments
    print('--'*40)
    for i, row in enumerate(responsesList):
        # Output some debugging info (prints every 10 responses)
        if((i+1) % 10 == 0 or i == 0 or (i+1) == len(responsesList)):
            print(f'Round 2: Getting counterargument {i+1} out of {len(responsesList)} (run #{n+1})...')

        # Get all standpoints that had been selected and were not selected by this LLM
        # this is a list of stand points the llm could have an argument against
        counterStandpointIndexes = []
        for standpointIndex, summaries in standpointSummaries.items():
            if(standpointIndex+1 != responsesList[i]['standpointNum'] and len(summaries) > 0):
                counterStandpointIndexes.append(standpointIndex)
        if (len(counterStandpointIndexes)==0):
            counterStandpointIndexes.append(responsesList[i]['standpointNum']-1)  # sometimes there is no disagreement!
        
        # Randomly select a standpoint to counter argue against
        counterStandpointIndex = np.random.choice(counterStandpointIndexes)

        # Randomly select a summary from the selected standpoint to counter
        opposingViewpoint = np.random.choice(standpointSummaries[counterStandpointIndex])
        
        # Then recover the round1 standpoint and reasoning
        standpoint=standpointOptions[responsesList[i]['standpointNum']-1]
        reasoning=responsesList[i]['reasoning']

        # Get this LLM's counterargument to this opposing viewpoint
        covdict = {key: row[key] for key in cov_keys if key in row} # extract just 'model' and covariates
        counterArgument = GetCounterArgument(covdict, topic, standpoint, reasoning, opposingViewpoint, n, force)

        # append the counterargument 
        responsesList[i]['opposingstandpoint']=counterStandpointIndex+1
        responsesList[i]['opposingviewpoint']=opposingViewpoint
        responsesList[i]['counterargument']=counterArgument
            
    # Post round 2: Summarize the counterarguments
    responsesDF = pd.DataFrame(responsesList)
    opposingStandpointSummaries = {} # {OpposingStandpointIndex: [summaries], ...}
    for i, opposingStandpoint in enumerate(standpointOptions):
        # Get all counterarguments using Pandas query and convert to list
        counterarguments = responsesDF.loc[responsesDF['opposingstandpoint'] == i+1, 'counterargument']
        counterarguments = list(counterarguments.values)
        # Get and save the counterargument summaries
        summaries = SummarizeResponses(topic, opposingStandpoint, counterarguments, True, n, force)
        opposingStandpointSummaries[i] = summaries
    allSummaries.append(opposingStandpointSummaries)

    # Round 3: Respond to counterarguments
    print('--'*40)
    for i, row in enumerate(responsesList):
        # Output some debugging info
        if((i+1) % 10 == 0 or i == 0 or (i+1) == len(responsesList)):
            print(f'Round 3: Getting response {i+1} out of {len(responsesList)} (run #{n+1})...')
        # print("Made it here!")
        # Select a counterargument randomly from those that opposed this LLM's viewpoint
        counterarguments = opposingStandpointSummaries[responsesList[i]['standpointNum'] - 1] # Get all counterarguments

        # If there is at least one counterargument, randomly choose one of them and get the LLM to respond to it
        if(len(counterarguments) > 0):
            counterargument = np.random.choice(counterarguments)
            standpoint=standpointOptions[responsesList[i]['standpointNum']-1]
            reasoning=responsesList[i]['reasoning']
            covdict = {key: row[key] for key in cov_keys if key in row} # extract just 'model' and covariates
            response = GetResponseToCounterArgument(covdict, topic, standpoint, reasoning, counterargument, n, force)
        # If there are no counterarguments to this LLM's standpoint, simply reuse the original reasoning from Round 1
        else:
            response = responsesList[i]['reasoning']
        if(len(counterarguments) > 0):
            responsesList[i]['counterargheard']=counterargument
        else:
            responsesList[i]['counterargheard']=responsesList[i]['reasoning']
        responsesList[i]['responsetocounter']=response

    # Post round 3: Summarize responses
    responsesDF = pd.DataFrame(responsesList)
    standpointSummarizedResponses = {} # {standpointNum: [summarizedResponses], ...}
    for i, standpoint in enumerate(standpointOptions):
        # Get all responses to counterarguments using Pandas query and convert to list
        responses = responsesDF.loc[responsesDF['standpointNum'] == i+1, 'responsetocounter']
        responses = list(responses.values)
        summaries = SummarizeResponses(topic, standpointOptions, responses, False, n, force)
        standpointSummarizedResponses[i]=summaries
    allSummaries.append(standpointSummarizedResponses)


    if(saveToFile):
        responsesDF.to_csv(directory_name+f'/responsesDF_n{n}.csv', index=False)
        with open(directory_name+f'/ResponseSummaries_n{n}.json', 'w', encoding="utf-8") as f:
            json.dump(allSummaries, f)

    # Return the summarized responses
    return standpointSummarizedResponses




def main():
    y=str(int(rng.integers(10**10)))
    directory_name = f'Results/trial_id_{y}'
    os.mkdir(directory_name)
    "Simulate a focus group using LLM agents"
    # SET USER PARAMS HERE
    models = [
        'gemini-2.5-flash',  # use
        'llama-3.3-70b-versatile', # use
        'qwen/qwen3-32b', # use
        #'deepseek-r1-distill-llama-70b',
        #'gemma2-9b-it',
        'openai/gpt-oss-120b', # use
    ]
    # the first entry of cov_keys must be 'model' and the remaining entries must be fields in twinpersonalities
    cov_keys=['model', 'gender', 'education', 'politics', 'pgreen']
    # enter the general topic (no more than several words); do not include punctuation at the end
    topic = 'the adoption of Nuclear Power. '
    # enter the standpoint options here; do not include punctuation at the end
    standpointOptions = [
        'implement rapid expansion of and investment in nuclear power',
        'maintain the current nuclear energy operations without change',
        'phase out existing nuclear plants and halt new construction',
        'prioritize researching improved implementations of nuclear power',
    ] 
    # enter any background info that will help the LLM to give realistic responses
    background = '''To help you choose an option, please know that according to the PEW Research Center, 
    59 percent of US residents favor expanding nuclear power. 73 percent of men 
    favor expanding it while only 44 percent of women favor expanding it. 
    69 percent of Republicans favor expanding compared to 52 percent of Democrats. 
    Environmentalists have mixed opinions, where some favor expanding it because 
    nuclear has a low-carbon footprint, while others are concerned about the risks of nuclear waste disposal.'''
    sampleSize = 3 # This is the sample size of LLM agents; set this for all experiments
    numRuns = 1 # Number of runs per experiment; usually 50-100 for traditional and 10-20 for Prytaneum
    saveToFile = True # Save the raw results of each experiment run to file; keep as True unless testing
    saveSimilaritiesToFile = True # Save the final cosine similarities to file
    force = True  # set to True if you don't want to use cached answers; caching allows you to re-run an experiment at no cost
    # END USER PARAMS

    # Initialize LLMs
    gemini.InitGoogleGemini()
    llm.InitGroq()

    # Initialize random seed
    np.random.seed(5591) # last four digits of Kevin's office number

    # Start the experiment for the given number of runs
    allStandpointResponses = {} # {standpointNum: [responsesRun1, responsesRun2, ...], ...}
    for n in range(numRuns):
        # Output debugging info
        if(n % 1 == 0):
            print('=='*40)
            print(f'Starting run #{n+1} out of {numRuns}...')

    # Start the experiment for the given number of runs
        standpointResponses = RunExperiment(models, cov_keys, topic, background, standpointOptions, directory_name,  
                        sampleSize, n, saveToFile, force)

        # Merge all intra-standpoint responses into a single string and save to list
        for standpointNum, responses in standpointResponses.items():
            # Initialize storage dict if this is the first run
            if(standpointNum not in allStandpointResponses):
                allStandpointResponses[standpointNum] = []
                #inspect_summaries[standpointNum] = []

            # Save the merged responses
            mergedResponses = ' '.join(responses)
            allStandpointResponses[standpointNum].append(mergedResponses)
            #inspect_summaries[standpointNum].append(mergedResponses)
        
    print('=='*40)
    print('Simulation completed.')

    # End of experiment: Get response similarity score

    # Initialize sentence transformer model
    transformerModel = SentenceTransformer('all-MiniLM-L6-v2')

    # Get the cosine similarity of each response to each other within each standpoint
    print('Calculating cosine similarities...')
    cosineSimilarities = [] # [cosineSimilarityOfStandpoint1, cosineSimilarityOfStandpoint2, ...}
    for standpointNum, responses in allStandpointResponses.items():
        # Ignore standpoints with no responses
        if(len(responses) == 0):
            continue

        # Get the cosine similarity of each response to each other response as a similarity matrix
        # (cosine similarity just measures how similar any statements are by turning the text into a numerical vector then computes a number from 1 to -1, -1 being opposite statements, 1 being basically the same)
        responsesEmbeddings = transformerModel.encode(responses)
        similarityMatrix = cosine_similarity(responsesEmbeddings)
        upperTriangle = np.triu_indices_from(similarityMatrix, k=1) # Get the upper triangle indices of the matrix, excluding the diagonal
        similarity = np.mean(similarityMatrix[upperTriangle])
        cosineSimilarities.append(similarity)

        # Output the full similarity matrix to .csv
        if(saveSimilaritiesToFile):
            np.savetxt(directory_name+f'/similarityMatrix_standpoint{standpointNum}.csv', similarityMatrix,
                delimiter=',', fmt='%.2f')
    
    # output parameters to text file for recordkeeping
    if (saveToFile):
        parameters_text=f'Trial ID number: {y}\n'
        parameters_text+='Models = '+str(models)+'\n'
        parameters_text+='Covariates = '+str(cov_keys)+'\n'
        parameters_text+='Topic = '+topic+'\n'
        parameters_text+='Standpoint Options = '+str(standpointOptions)+'\n'
        parameters_text+='Background = '+background+'\n'
        parameters_text+='Sample Size = '+str(sampleSize)+'\n'
        parameters_text+='Number of Experimental Runs = '+str(numRuns)
        with open(directory_name+'/parameter_settings.txt', 'w') as f:
            f.write(parameters_text)


    print('Done.')
    



    # End of experiment

if __name__ == '__main__':
    main()


