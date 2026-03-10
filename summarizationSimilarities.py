
import os
import json
import pandas as pd
from sentence_transformers import SentenceTransformer, util
transformerModel = SentenceTransformer('all-MiniLM-L6-v2')

filename = os.getcwd() + '\\Results\\trial_id_6737109129\\ResponsesDF_n0.csv'
tempdf = pd.read_csv(filename)
responseDict = tempdf.to_dict(orient='list')


def AnalyzeSimulated():
    filename = os.getcwd() + '\\Results\\trial_id_6737109129\\ResponseSummaries_n0.json'
    with open(filename, 'r', encoding="utf-8") as f:
            summaries = f.read()
    standpointSummaries=json.loads(summaries)

    summsStandpointNumEmbeddings = {} # {standpointNum: [summEmbeddings], ...}
    for standpointNum, summarizations in standpointSummaries[0].items():  # using the first-round summaries hence the index 0
        summsStandpointNumEmbeddings[standpointNum] = []
        for summ in summarizations:
            summsStandpointNumEmbeddings[standpointNum].append(transformerModel.encode(summ))

    # Initialize keys to store the cosine similarities
    # We need 5 columns for similarity since there are 5 summarizations for each standpoint
    responseDict['similarityColumn1'] = []
    responseDict['similarityColumn2'] = []
    responseDict['similarityColumn3'] = []
    responseDict['similarityColumn4'] = []
    responseDict['similarityColumn5'] = []
    

    # Record the similarity between the reasonings and each summarization within their standpoint
    print('Calculating cosine similarities...')
    for standpointNum, reasoning in zip(responseDict['standpointNum'], responseDict['reasoning']):
        # Calculate and save cosine similarities between this response and each summarization
        # util.cos_sim() returns a tensor of size 1, so fetch the result, convert it to float, and round to 4 decimal points
        responseEmbeddings = transformerModel.encode(reasoning)
        standpointNum=str(standpointNum-1)
        responseDict['similarityColumn1'].append(round(float(util.cos_sim(responseEmbeddings, summsStandpointNumEmbeddings[standpointNum][0])[0][0]), 4))
        responseDict['similarityColumn2'].append(round(float(util.cos_sim(responseEmbeddings, summsStandpointNumEmbeddings[standpointNum][1])[0][0]), 4))
        responseDict['similarityColumn3'].append(round(float(util.cos_sim(responseEmbeddings, summsStandpointNumEmbeddings[standpointNum][2])[0][0]), 4))
        responseDict['similarityColumn4'].append(round(float(util.cos_sim(responseEmbeddings, summsStandpointNumEmbeddings[standpointNum][3])[0][0]), 4))
        responseDict['similarityColumn5'].append(round(float(util.cos_sim(responseEmbeddings, summsStandpointNumEmbeddings[standpointNum][4])[0][0]), 4))

    # Format everything as a dataframe and save to .csv
    filename = os.getcwd() + '\\Results\\trial_id_6737109129\\ResponsesDF_similarities_n0.csv'
    print('Saving results to file:', filename)
    df = pd.DataFrame(responseDict)
    df.to_csv(filename, index=False)
    print('Done.')


def main():
    print('=='*40)
    print('Starting simulated analysis...')
    AnalyzeSimulated()

if __name__ == '__main__':
    main()

