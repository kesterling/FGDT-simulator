# https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500

import os
import sys # Module that helps with file in
sys.path.append('../')
import pandas as pd # Used for data descriptions + manipulation
from datasets import load_dataset

current_directory = os.getcwd()
print(f"Current working directory: {current_directory}")



# load digital twins data (N=2058) from Hugging Face Hub:
ds = load_dataset("LLM-Digital-Twin/Twin-2K-500", "full_persona")
# wave_split = load_dataset("LLM-Digital-Twin/Twin-2K-500", "wave_split")
data=pd.DataFrame(ds["data"])
# print(data.persona_summary[0])

# initialize the digital twins personalities dictionary
personalities_dict = {
        'region': [],
        'gender': [],
        'age': [],
        'education': [],
        'race': [],
        'uscitizen': [],
        'maritalstatus': [],
        'religion': [],
        'religionattend': [],
        'partyid': [],
        'income': [],
        'politics': [],
        'hhsize': [],
        'employstatus': [],
        'extraversion': [],
        'pextraversion': [],
        'agreeableness': [],
        'pagreeableness': [],
        'conscientiousness': [],
        'pconscientiousness': [],
        'openness': [],
        'popenness': [],
        'neuroticism': [],
        'pneuroticism': [],
        'needforcognition': [],
        'pneedforcognition': [],
        'agency': [],
        'pagency': [],
        'communion': [],
        'pcommunion': [],
        'minimalism': [],
        'pminimalism': [],
        'bes': [],
        'pbes': [],
        'green': [],
        'pgreen': [],
        'crt2': [],
        'pcrt2': [],
        'fluid': [],
        'pfluid': [],
        'crystallized': [],
        'pcrystallized': [],
        'syllogism': [],
        'psyllogism': [],
        'actual': [],
        'pactual': [],
        'overconfidence': [],
        'poverconfidence': [],
        'overplacement': [],
        'poverplacement': [],
        'ultimatum_sender': [],
        'pultimatum_sender': [],
        'ultimatum_accepted': [],
        'pultimatum_accepted': [],
        'mentalaccounting': [],
        'pmentalaccounting': [],
        'socialdesirability': [],
        'psocialdesirability': [],
        'conscientiousness_w2': [],
        'pconscientiousness_w2': [],
        'anxiety': [],
        'panxiety': [],
        'hi': [],
        'phi': [],
        'hc': [],
        'phc': [],
        'vi': [],
        'pvi': [],
        'vc': [],
        'pvc': [],
        'finliteracy': [],
        'pfinliteracy': [],
        'numeracy': [],
        'pnumeracy': [],
        'deductive': [],
        'pdeductive': [],
        'forwardflow': [],
        'pforwardflow': [],
        'discount': [],
        'pdiscount': [],
        'presentbias': [],
        'ppresentbias': [],
        'riskaversion': [],
        'priskaversion': [],
        'lossaversion': [],
        'plossaversion': [],
        'trustgame_sender': [],
        'ptrustgame_sender': [],
        'trustgame_receiver': [],
        'ptrustgame_receiver': [],
        'rfs': [],
        'prfs': [],
        'sttw': [],
        'psttw': [],
        'depression': [],
        'pdepression': [],
        'cnfu': [],
        'pcnfu': [],
        'selfmonitor': [],
        'pselfmonitor': [],
        'scc': [],
        'pscc': [],
        'needforclosure': [],
        'pneedforclosure': [],
        'maximization': [],
        'pmaximization': [],
        'wason': [],
        'pwason': [],
        'dictator_sender': [],
        'pdictator_sender': []
    }


demokeys_list=['region', 'gender', 'age', 'education', 'race', 'uscitizen', 'maritalstatus', 'religion', 'religionattend', 'partyid', 'income', 'politics', 'hhsize', 'employstatus']

psyckeys_list=['extraversion', 'agreeableness', 'conscientiousness', 'openness', 'neuroticism', 'needforcognition', 'agency', 'communion', 'minimalism', 'bes', 'green', 'crt2', 'fluid', 'crystallized', 'syllogism', 'actual', 'overconfidence', 'overplacement', 'ultimatum_sender', 'ultimatum_accepted', 'mentalaccounting', 'socialdesirability', 'conscientiousness_w2', 'anxiety', 'hi', 'hc', 'vi', 'vc', 'finliteracy', 'numeracy', 'deductive', 'forwardflow', 'discount', 'presentbias', 'riskaversion', 'lossaversion', 'trustgame_sender', 'trustgame_receiver', 'rfs', 'sttw', 'depression', 'cnfu', 'selfmonitor', 'scc', 'needforclosure', 'maximization', 'wason', 'dictator_sender']

ppsyckeys_list=['pextraversion', 'pagreeableness', 'pconscientiousness', 'popenness', 'pneuroticism', 'pneedforcognition', 'pagency', 'pcommunion', 'pminimalism', 'pbes', 'pgreen', 'pcrt2', 'pfluid', 'pcrystallized', 'psyllogism', 'pactual', 'poverconfidence', 'poverplacement', 'pultimatum_sender', 'pultimatum_accepted', 'pmentalaccounting', 'psocialdesirability', 'pconscientiousness_w2', 'panxiety', 'phi', 'phc', 'pvi', 'pvc', 'pfinliteracy', 'pnumeracy', 'pdeductive', 'pforwardflow', 'pdiscount', 'ppresentbias', 'priskaversion', 'plossaversion', 'ptrustgame_sender', 'ptrustgame_receiver', 'prfs', 'psttw', 'pdepression', 'pcnfu', 'pselfmonitor', 'pscc', 'pneedforclosure', 'pmaximization', 'pwason', 'pdictator_sender']

demofields_list=["Geographic region: ", "Gender: ", "Age: ", "Education level: ", "Race: ", "Citizen of the US: ", "Marital status: ", "Religion: ", "Religious attendance: ", "Political affiliation: ", "Income: ", "Political views: ", "Household size: ", "Employment status: "]

psycfields_list=["score_extraversion = ", "score_agreeableness = ", "score_conscientiousness = ", "score_openness = ", "score_neuroticism = ", "score_needforcognition = ", "score_agency = ", "score_communion = ", "score_minimalism = ", "score_BES = ", "score_GREEN = ", "crt2_score = ", "score_fluid = ", "score_crystallized = ", "score_syllogism_merged = ", "score_actual_total = ", "score_overconfidence = ", "score_overplacement = ", "score_ultimatum_sender = ", "score_ultimatum_accepted = ", "score_mentalaccounting = ", "score_socialdesirability = ", "wave2_score_conscientiousness = ", "score_anxiety = ", "score_HI = ", "score_HC = ", "score_VI = ", "score_VC = ", "score_finliteracy = ", "score_numeracy = ", "score_deductive_certainty = ", "score_forwardflow = ", "score_discount = ", "score_presentbias = ", "score_riskaversion = ", "score_lossaversion = ", "score_trustgame_sender = ", "score_trustgame_receiver = ", "score_RFS = ", "score_ST-TW = ", "score_depression = ", "score_CNFU-S = ", "score_selfmonitor = ", "score_SCC = ", "score_needforclosure = ", "score_maximization = ", "score_wason = ", "score_dictator_sender = "]


# append each twin's values to construct the dictionary
for i in range(len(data.persona_summary)):
    # print(i)
    for j in range(len(demofields_list)):
        begin=data.persona_summary[i].find(demofields_list[j])
        if begin > -1:
            end=data.persona_summary[i].find(demofields_list[j])+len(demofields_list[j])
            if demofields_list[j] == "Geographic region: ":
                personalities_dict[demokeys_list[j]].append(data.persona_summary[i][end:end+data.persona_summary[i][end:].find(" ")])
            else:
                personalities_dict[demokeys_list[j]].append(data.persona_summary[i][end:end+data.persona_summary[i][end:].find("\n")])
            if len(data.persona_summary[i][end:end+data.persona_summary[i][end:].find("\n")])==0:
                print(i, data.persona_summary[i][begin:end], " ****Error****")
        else:
            personalities_dict[demokeys_list[j]].append("None")

    for j in range(len(psycfields_list)):
        begin=data.persona_summary[i].find(psycfields_list[j])
        if begin > -1:
            end=data.persona_summary[i].find(psycfields_list[j])+len(psycfields_list[j])
            personalities_dict[psyckeys_list[j]].append(data.persona_summary[i][end:end+data.persona_summary[i][end:].find(" ")])
            personalities_dict[ppsyckeys_list[j]].append(data.persona_summary[i][end+data.persona_summary[i][end:].find("(")+1:end+data.persona_summary[i][end:].find(" percentile")-2])
            if len(data.persona_summary[i][end:end+data.persona_summary[i][end:].find(" ")])==0:
                print(i, data.persona_summary[i][begin:end], " ****Error****")
        else:
            personalities_dict[psyckeys_list[j]].append("None")
            personalities_dict[ppsyckeys_list[j]].append("None")
    

# not including the concept of self open-ended responses


twin_personalities=pd.DataFrame(personalities_dict)

# count up the number of missing values
k=0
for i in range(len(twin_personalities)):
    for j in twin_personalities.loc[i]:
        if j=="None":
            k=k+1



twin_personalities.to_csv('twin_personalities_file.csv', index=False)

