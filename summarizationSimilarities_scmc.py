import GoogleGemini as gemini
import pandas as pd
from sentence_transformers import SentenceTransformer, util

def SummarizeResponses(responses: list, topic: str, supporting=True, maxSummarizations=5, force=False) -> list[str]:
    """Summarize all given responses that justify the given standpoint.
    Set counterargument to True if the responses are instead arguing against the given standpoint.
    Set standpoint as a list of standpoints to not summarize in terms of any particular standpoint."""
    if(len(responses) < 1):
        return []

    prompt = f'You are a helpful assistant tasked with summarizing argumentative statements on the topic of {topic}. '
    prompt += 'I have gathered a list of statements that participants in an online focus group platform have provided as their reasoning for being '
    if(supporting):
        prompt += f'in support of {topic}. '
    else:
        prompt += f'against {topic}. '
    
    if(maxSummarizations == 1):
        prompt += 'Summarize the following statements as exactly one concise, single-sentence summary. '
        prompt += 'Your summary should be worded as passive tone viewpoints, so that they capture '
        prompt += 'as many viewpoints presented in the given statements as possible. '

        prompt += 'Start the summary on a line with a dash.'
    else:
        prompt += f'Summarize the following statements as no more than {maxSummarizations} concise summaries. '
        prompt += 'Your summaries should be worded as passive tone viewpoints, so that they capture '
        prompt += 'as many viewpoints presented in the given statements as possible. '

        prompt += 'Use as few summaries as are necessary to achieve this. '
        prompt += 'Start each summary on a new line with a dash.'
    prompt += '\n\n'

    prompt += 'The statements are:\n'
    for i, post in enumerate(responses):
        prompt += f'Statement {i+1}: "{post}"\n'
    prompt += '\n'
        
    prompt += 'Your summarized viewpoints:\n'

    response = gemini.AskGoogleGemini(prompt, force=force)
    response = response.replace('\n', '')
    response = response.split('- ')
    response = response[1:]
    return response

def main():
    "Get real data from Euchan's town hall and calculate cosine similarities between responses and summarizations."
    columnGroupings = 'pst_house_size'
    topic = 'increasing the size of the U.S. Congress House of Representatives'

    userIds = []
    ethnicities = []
    genders = []
    educations = []
    parties = []
    topics = []
    supports = []
    similarities1 = []
    similarities2 = []
    similarities3 = []
    similarities4 = []
    similarities5 = []
    similarities6 = []
    similarities7 = []
    similarities8 = []
    similarities9 = []

    print('Initializing...')
    gemini.InitGoogleGemini()

    transformerModel = SentenceTransformer('all-MiniLM-L6-v2')
    print('--'*40)

    print('Fetching SCMC town hall data...')
    infilename = 'SCMC_data/scmc_data.csv'
    df = pd.read_csv(infilename)

    columnReasonings = columnGroupings + '_expln'

    df = df[df[columnGroupings].notna()]
    df = df[df[columnReasonings].str.len() >= 10]
    df = df[['id', 'race_ethnicity', 'gender', 'education', 'party3', columnGroupings, columnReasonings]]
    df = df.dropna()

    responsesAgainst = list(df[df[columnGroupings] == 0][columnReasonings])
    responsesSupport = list(df[df[columnGroupings] == 1][columnReasonings])
    print(f'Fetched {len(df)} responses with reasoning length >= 10 letters.')
    print(f'Number responses against: {len(responsesAgainst)}. In support: {len(responsesSupport)}.')

    print('Getting summarizations against...')
    summarizationsAgainst = []
    for maxSummarizations in [1, 3, 5]:
        summarizations = SummarizeResponses(responsesAgainst, topic, supporting=False, maxSummarizations=maxSummarizations)
        summarizationsAgainst.extend(summarizations)

    print('Getting summarizations in support...')
    summarizationsSupport = []
    for maxSummarizations in [1, 3, 5]:
        summarizations = SummarizeResponses(responsesSupport, topic, supporting=True, maxSummarizations=maxSummarizations)
        summarizationsSupport.extend(summarizations)
    
    summsAgainstEmbeddings = []
    summsSupportEmbeddings = []
    for (summAgainst, summSupport) in zip(summarizationsAgainst, summarizationsSupport):
        summsAgainstEmbeddings.append(transformerModel.encode(summAgainst))
        summsSupportEmbeddings.append(transformerModel.encode(summSupport))

    print('Calculating cosine similarities for responses against...')
    for index, row in df[df[columnGroupings] == 0].iterrows():
        userIds.append(row['id'])
        ethnicities.append(row['race_ethnicity'])
        genders.append(row['gender'])
        educations.append(row['education'])
        parties.append(row['party3'])
        topics.append('Increasing size of House of Representatives')
        supports.append(False)

        responseEmbeddings = transformerModel.encode(row[columnReasonings])
        similarities1.append(round(float(util.cos_sim(responseEmbeddings, summsAgainstEmbeddings[0])[0][0]), 4))
        similarities2.append(round(float(util.cos_sim(responseEmbeddings, summsAgainstEmbeddings[1])[0][0]), 4))
        similarities3.append(round(float(util.cos_sim(responseEmbeddings, summsAgainstEmbeddings[2])[0][0]), 4))
        similarities4.append(round(float(util.cos_sim(responseEmbeddings, summsAgainstEmbeddings[3])[0][0]), 4))
        similarities5.append(round(float(util.cos_sim(responseEmbeddings, summsAgainstEmbeddings[4])[0][0]), 4))
        similarities6.append(round(float(util.cos_sim(responseEmbeddings, summsAgainstEmbeddings[5])[0][0]), 4))
        similarities7.append(round(float(util.cos_sim(responseEmbeddings, summsAgainstEmbeddings[6])[0][0]), 4))
        similarities8.append(round(float(util.cos_sim(responseEmbeddings, summsAgainstEmbeddings[7])[0][0]), 4))
        similarities9.append(round(float(util.cos_sim(responseEmbeddings, summsAgainstEmbeddings[8])[0][0]), 4))

    print('Calculating cosine similarities for responses in support...')
    for index, row in df[df[columnGroupings] == 1].iterrows():
        userIds.append(row['id'])
        ethnicities.append(row['race_ethnicity'])
        genders.append(row['gender'])
        educations.append(row['education'])
        parties.append(row['party3'])
        topics.append('Increasing size of House of Representatives')
        supports.append(True)

        responseEmbeddings = transformerModel.encode(row[columnReasonings])
        similarities1.append(round(float(util.cos_sim(responseEmbeddings, summsSupportEmbeddings[0])[0][0]), 4))
        similarities2.append(round(float(util.cos_sim(responseEmbeddings, summsSupportEmbeddings[1])[0][0]), 4))
        similarities3.append(round(float(util.cos_sim(responseEmbeddings, summsSupportEmbeddings[2])[0][0]), 4))
        similarities4.append(round(float(util.cos_sim(responseEmbeddings, summsSupportEmbeddings[3])[0][0]), 4))
        similarities5.append(round(float(util.cos_sim(responseEmbeddings, summsSupportEmbeddings[4])[0][0]), 4))
        similarities6.append(round(float(util.cos_sim(responseEmbeddings, summsSupportEmbeddings[5])[0][0]), 4))
        similarities7.append(round(float(util.cos_sim(responseEmbeddings, summsSupportEmbeddings[6])[0][0]), 4))
        similarities8.append(round(float(util.cos_sim(responseEmbeddings, summsSupportEmbeddings[7])[0][0]), 4))
        similarities9.append(round(float(util.cos_sim(responseEmbeddings, summsSupportEmbeddings[8])[0][0]), 4))
    print('--'*40)
    
    filename = 'Results/summarizationSimilarities_SCMC.csv'
    print('Saving results to file:', filename)
    df = pd.DataFrame({
        'User ID': userIds,
        'Ethnicity': ethnicities,
        'Gender': genders,
        'Education': educations,
        'Party': parties,
        'Topic': topics,
        'In Support': supports,
        'Similarity1': similarities1,
        'Similarity2': similarities2,
        'Similarity3': similarities3,
        'Similarity4': similarities4,
        'Similarity5': similarities5,
        'Similarity6': similarities6,
        'Similarity7': similarities7,
        'Similarity8': similarities8,
        'Similarity9': similarities9,
    })
    df.to_csv(filename, index=False)
    print('Done.')

if __name__ == '__main__':
    main()