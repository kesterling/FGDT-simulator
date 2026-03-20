
## full listing of keys in digitaltwins dataset:
# demokeys_list=['region', 'gender', 'age', 'education', 'race', 'uscitizen', 'maritalstatus', 'religion', 'religionattend', 'partyid', 'income', 'politics', 'hhsize', 'employstatus']
# psyckeys_list=['extraversion', 'agreeableness', 'conscientiousness', 'openness', 'neuroticism', 'needforcognition', 'agency', 'communion', 'minimalism', 'bes', 'green', 'crt2', 'fluid', 'crystallized', 'syllogism', 'actual', 'overconfidence', 'overplacement', 'ultimatum_sender', 'ultimatum_accepted', 'mentalaccounting', 'socialdesirability', 'conscientiousness_w2', 'anxiety', 'hi', 'hc', 'vi', 'vc', 'finliteracy', 'numeracy', 'deductive', 'forwardflow', 'discount', 'presentbias', 'riskaversion', 'lossaversion', 'trustgame_sender', 'trustgame_receiver', 'rfs', 'sttw', 'depression', 'cnfu', 'selfmonitor', 'scc', 'needforclosure', 'maximization', 'wason', 'dictator_sender']
# ppsyckeys_list=['pextraversion', 'pagreeableness', 'pconscientiousness', 'popenness', 'pneuroticism', 'pneedforcognition', 'pagency', 'pcommunion', 'pminimalism', 'pbes', 'pgreen', 'pcrt2', 'pfluid', 'pcrystallized', 'psyllogism', 'pactual', 'poverconfidence', 'poverplacement', 'pultimatum_sender', 'pultimatum_accepted', 'pmentalaccounting', 'psocialdesirability', 'pconscientiousness_w2', 'panxiety', 'phi', 'phc', 'pvi', 'pvc', 'pfinliteracy', 'pnumeracy', 'pdeductive', 'pforwardflow', 'pdiscount', 'ppresentbias', 'priskaversion', 'plossaversion', 'ptrustgame_sender', 'ptrustgame_receiver', 'prfs', 'psttw', 'pdepression', 'pcnfu', 'pselfmonitor', 'pscc', 'pneedforclosure', 'pmaximization', 'pwason', 'pdictator_sender']

# This is the listing of keys for the available prompt blocks, and also must include 'model'
def existing_prompt_blocks():
    existing_prompt_blocks=['model', 'region', 'gender', 'age', 'education', 'race', 'uscitizen', 'maritalstatus', 'religion', 'religionattend', 'partyid', 'income', 'politics', 'hhsize', 'employstatus', 
                            'pextraversion', 'pagreeableness', 'pconscientiousness', 'popenness', 'pneuroticism', 'pneedforcognition', 'pagency', 'pcommunion', 'pminimalism', 'pbes', 'pgreen', 
                            'pcrt2', 'pfluid', 'pcrystallized', 'psyllogism', 'pactual', 'poverconfidence', 'poverplacement', 'pultimatum_sender', 'pultimatum_accepted', 'pmentalaccounting', 'psocialdesirability', 
                            'pconscientiousness_w2', 'panxiety', 'phi', 'phc', 'pvi', 'pvc', 'pfinliteracy', 'pnumeracy', 'pdeductive', 'pforwardflow', 'pdiscount', 'ppresentbias', 'priskaversion', 'plossaversion', 
                            'ptrustgame_sender', 'ptrustgame_receiver', 'prfs', 'psttw', 'pdepression', 'pcnfu', 'pselfmonitor', 'pscc', 'pneedforclosure', 'pmaximization', 'pwason', 'pdictator_sender']
    return existing_prompt_blocks


def digitaltwins_blocks(covdict: dict, topic: str):
    # Each covariate block must have the following structure:
    # if 'keyname' in covdict.keys():
    #    if covdict['keyname']=="None":
    #        pass
    #    # elif block optional -- used for classification for keynames that are floats
    #    else: 
    #        prompt += f'prompt text {covdict['keyname']} etc.'
    prompt = 'You will adopt the personality of a '
    if 'education' in covdict.keys():
        if covdict['education']=="None":
            pass
        else:
            prompt += f'{covdict['education']}-educated, '
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
    if 'region' in covdict.keys():
        if covdict['region']=="None":
            pass
        else:
            prompt += f'You come from the {covdict['region']} region of the United States. '
    if 'age' in covdict.keys():
        if covdict['age']=="None":
            pass
        else:
            prompt += f'You are {covdict['age']} years old. '
    if 'uscitizen' in covdict.keys():
        if covdict['uscitizen']=="None":
            pass
        elif covdict['uscitizen']=="Yes":
            prompt += 'You are a citizen of the United States. '
        else:
            prompt += 'You are not a citizen of the United States. '
    if 'maritalstatus' in covdict.keys():
        if covdict['maritalstatus']=="None":
            pass
        else:
            prompt += f'Your marital status is {covdict['maritalstatus']}. '
    if 'religion' in covdict.keys():
        if covdict['religion']=="None":
            pass
        else:
            prompt += f'Your religious affiliation is {covdict['religion']}. '
    if 'religionattend' in covdict.keys():
        if covdict['religionattend']=="None":
            pass
        else:
            prompt += f'You {covdict['religionattend']} attend religious services. '
    if 'partyid' in covdict.keys():
        if covdict['partyid']=="None":
            pass
        else:
            prompt += f'Your political party identification is {covdict['partyid']}. '
    if 'income' in covdict.keys():
        if covdict['income']=="None":
            pass
        else:
            prompt += f'Your income is {covdict['income']}. '
    if 'hhsize' in covdict.keys():
        if covdict['hhsize']=="None":
            pass
        else:
            prompt += f'Your household size is {covdict['hhsize']}. '
    if 'employstatus' in covdict.keys():
        if covdict['employstatus']=="None":
            pass
        else:
            prompt += f'Your employment status is {covdict['employstatus']}. '
    if 'pextraversion' in covdict.keys():
        if covdict['pextraversion']=="None":
            pass
        elif float(covdict['pextraversion']) <= 33:
            prompt += 'You are introverted. '
        elif float(covdict['pextraversion']) > 33 and float(covdict['pextraversion']) <= 67:
            pass
            # Not sure how to describe someone in the middle 
            # Let's leave implicit for all personality covariates -- the issue is that simply mentioning the trait can prime the LLM
            # prompt += 'You are neither introverted nor extraverted. '
        else:
            prompt += 'You are extraverted. '
    if 'pagreeableness' in covdict.keys():
        if covdict['pagreeableness']=="None":
            pass
        elif float(covdict['pagreeableness']) <= 33:
            prompt += 'You have a low level of agreeableness and compassion for others. '
        elif float(covdict['pagreeableness']) > 33 and float(covdict['pagreeableness']) <= 67:
            pass
            # prompt += 'You are moderately agreeable. '
        else:
            prompt += 'You have a high level of agreeableness and compassion for others. '
    if 'pconscientiousness' in covdict.keys():
        if covdict['pconscientiousness']=="None":
            pass
        elif float(covdict['pconscientiousness']) <= 33:
            prompt += 'You have a low level of conscientiousness, with low self-discipline or goal-directed behavior. '
        elif float(covdict['pconscientiousness']) > 33 and float(covdict['pconscientiousness']) <= 67:
            pass
            # prompt += 'You have a moderate level of conscientiousness. '
        else:
            prompt += 'You have a high level of conscientiousness, with high self-discipline and goal-directed behavior. '
    if 'popenness' in covdict.keys():
        if covdict['popenness']=="None":
            pass
        elif float(covdict['popenness']) <= 33:
            prompt += 'You have a low level of openness, with low curiosity or receptiveness to new experiences. '
        elif float(covdict['popenness']) > 33 and float(covdict['popenness']) <= 67:
            pass
            # prompt += 'You have a moderate level of openness. '
        else:
            prompt += 'You have a high level of openness, with lots of curiosity and receptiveness to new experiences. '
    if 'pneuroticism' in covdict.keys():
        if covdict['pneuroticism']=="None":
            pass
        elif float(covdict['pneuroticism']) <= 33:
            prompt += 'You are low on the neuroticism scale, meaning you are emotionally stable. '
        elif float(covdict['pneuroticism']) > 33 and float(covdict['pneuroticism']) <= 67:
            pass
            # prompt += 'You have a modest level of neuroticism. '
        else:
            prompt += 'You are high on the neuroticism scale, meaning you are emotionally unstable. '
    if 'pneedforcognition' in covdict.keys():
        if covdict['pneedforcognition']=="None":
            pass
        elif float(covdict['pneedforcognition']) <= 33:
            prompt += 'You have a low need for cognition, meaning you do not seek out or enjoy complex cognitive tasks. '
        elif float(covdict['pneedforcognition']) > 33 and float(covdict['pneedforcognition']) <= 67:
            prompt += 'You have a moderate level of need for cognition, meaning you seek out and enjoy complex cognitive tasks. '
        else:
            prompt += 'You have high need for cognition. '
    if 'pagency' in covdict.keys():
        if covdict['pagency']=="None":
            pass
        elif float(covdict['pagency']) <= 33:
            prompt += 'You have a low level of agency, meaning you do not see yourself as self-advancing within social hierarchies. '
        elif float(covdict['pagency']) > 33 and float(covdict['pagency']) <= 67:
            pass
            # prompt += 'You have a moderate level of agency. '
        else:
            prompt += 'You have a high level of agency, meaning you see yourself as self-advancing within social hierarchies. '
    if 'pcommunion' in covdict.keys():
        if covdict['pcommunion']=="None":
            pass
        elif float(covdict['pcommunion']) <= 33:
            prompt += 'You have a low level of communion, which means you do not prioritize maintaining positive relationships with others. '
        elif float(covdict['pcommunion']) > 33 and float(covdict['pcommunion']) <= 67:
            pass
            # prompt += 'You have a moderate amount of comunioni. '
        else:
            prompt += 'You have a high level of communion, meaning you prioritize maintaining positive relationships with others. '
    if 'pminimalism' in covdict.keys():
        if covdict['pminimalism']=="None":
            pass
        elif float(covdict['pminimalism']) <= 33:
            prompt += 'You do not prefer minimialism. '
        elif float(covdict['pminimalism']) > 33 and float(covdict['pminimalism']) <= 67:
            pass
            # prompt += 'You have a moderate level of minimalism. '
        else:
            prompt += 'You have a strong preference for minimalism. '
    if 'pbes' in covdict.keys():
        if covdict['pbes']=="None":
            pass
        elif float(covdict['pbes']) <= 33:
            prompt += 'You have a low level of empathy. '
        elif float(covdict['pbes']) > 33 and float(covdict['pbes']) <= 67:
            pass
            # prompt += 'You have a moderate amount of empathy. '
        else:
            prompt += 'You have a high level of empathy. '
    if 'pgreen' in covdict.keys():
        if covdict['pgreen']=="None":
            pass
        elif float(covdict['pgreen']) <= 33:
            prompt += 'You have low affinity for environmentalism. '
        elif float(covdict['pgreen']) > 33 and float(covdict['pgreen']) <= 67:
            prompt += 'You are neutral regarding environmentalism. '
        else:
            prompt += 'You have high affinity for environmentalism. '
    if 'pcrt2' in covdict.keys():
        if covdict['pcrt2']=="None":
            pass
        elif float(covdict['pcrt2']) <= 33:
            prompt += 'You have a low ability to suppress an intuitive (System 1) wrong answer. You do not tend favor a reflective and deliberative (System 2) right answer. '
        elif float(covdict['pcrt2']) > 33 and float(covdict['pcrt2']) <= 67:
            pass
            # prompt += ' '
        else:
            prompt += 'You have a high ability to suppress an intuitive (System 1) wrong answer in favor of a reflective and deliberative (System 2) right answer. '
    if 'pfluid' in covdict.keys():
        if covdict['pfluid']=="None":
            pass
        elif float(covdict['pfluid']) <= 33:
            prompt += 'You have little ability to reason and solve novel problems independent of prior knowledge. '
        elif float(covdict['pfluid']) > 33 and float(covdict['pfluid']) <= 67:
            pass
        # prompt += ' '
        else:
            prompt += 'You have a high ability to reason and solve novel problems independent of prior knowledge. '
    if 'pcrystallized' in covdict.keys():
        if covdict['pcrystallized']=="None":
            pass
        elif float(covdict['pcrystallized']) <= 33:
            prompt += 'You have low crystallized intelligence, meaning you have not acquired signficant knowledge and skills through experience or education. '
        elif float(covdict['pcrystallized']) > 33 and float(covdict['pcrystallized']) <= 67:
            pass
        # prompt += ' '
        else:
            prompt += 'You have high crystallized intelligence, meaning you have acquired signficant knowledge and skills through experience and education. '
    if 'psyllogism' in covdict.keys():
        if covdict['psyllogism']=="None":
            pass
        elif float(covdict['psyllogism']) <= 33:
            prompt += 'You have a low ability to solve verbal reasoning problems. '
        elif float(covdict['psyllogism']) > 33 and float(covdict['psyllogism']) <= 67:
            pass
        # prompt += ' '
        else:
            prompt += 'You have a high ability to solve verbal reasoning problems. '
    if 'pactual' in covdict.keys():
        if covdict['pactual']=="None":
            pass
        elif float(covdict['pactual']) <= 33:
            prompt += 'You scored low on a logic and intelligence test. '
        elif float(covdict['pactual']) > 33 and float(covdict['pactual']) <= 67:
            pass
        # prompt += ' '
        else:
            prompt += 'You scored high on a logic and intelligence test. '
    if 'poverconfidence' in covdict.keys():
        if covdict['poverconfidence']=="None":
            pass
        elif float(covdict['poverconfidence']) <= 33:
            prompt += 'When you took an intelligence test, your prediction of what you would score was below what you actually scored. '
        elif float(covdict['poverconfidence']) > 33 and float(covdict['poverconfidence']) <= 67:
            prompt += 'When you took an intelligence test, your prediction of what you would score was about equal to what you actually scored. '
        else:
            prompt += 'When you took an intelligence test, your prediction of what you would score was above what you actually scored. '
    if 'poverplacement' in covdict.keys():
        if covdict['poverplacement']=="None":
            pass
        elif float(covdict['poverplacement']) <= 33:
            prompt += 'When you took an intelligence test, your prediction of what you would score was below what you believed others would score. '
        elif float(covdict['poverplacement']) > 33 and float(covdict['poverplacement']) <= 67:
            prompt += 'When you took an intelligence test, your prediction of what you would score was about equal to what you believed others would score. '
        else:
            prompt += 'When you took an intelligence test, your prediction of what you would score was above what you believed others would score. '
    if 'pultimatum_sender' in covdict.keys():
        if covdict['pultimatum_sender']=="None":
            pass
        elif float(covdict['pultimatum_sender']) <= 33:
            prompt += 'When you played the ultimatum game, you tended to send a small amount of money to your opponent. '
        elif float(covdict['pultimatum_sender']) > 33 and float(covdict['pultimatum_sender']) <= 67:
            pass
            prompt += ' '
        else:
            prompt += 'When you played the ultimatum game, you tended to send a large amount of money to your opponent. '
    if 'pultimatum_accepted' in covdict.keys():
        if covdict['pultimatum_accepted']=="None":
            pass
        elif float(covdict['pultimatum_accepted']) <= 33:
            prompt += 'When you played the ultimatum game, you had a low percentage for the number of offers you accepted. '
        elif float(covdict['pultimatum_accepted']) > 33 and float(covdict['pultimatum_accepted']) <= 67:
            pass
            # prompt += ' '
        else:
            prompt += 'When you played the ultimatum game, you had a high percentage for the number of offers you accepted. '
    if 'pmentalaccounting' in covdict.keys():
        if covdict['pmentalaccounting']=="None":
            pass
        elif float(covdict['pmentalaccounting']) <= 33:
            prompt += 'You scored low on a test for your ability to do mental accounting. '
        elif float(covdict['pmentalaccounting']) > 33 and float(covdict['pmentalaccounting']) <= 67:
            pass
            # prompt += ' '
        else:
            prompt += 'You scored high on a test for your ability to do mental accounting. '
    if 'psocialdesirability' in covdict.keys():
        if covdict['psocialdesirability']=="None":
            pass
        elif float(covdict['psocialdesirability']) <= 33:
            prompt += 'You tend to respond to questions in a truthful way rather than a socially desirable way. '
        elif float(covdict['psocialdesirability']) > 33 and float(covdict['psocialdesirability']) <= 67:
            pass
            # prompt += ' '
        else:
            prompt += 'You tend to respond to questions in a socially desirable way rather than a truthful way. '
    if 'pconscientiousness_w2' in covdict.keys():
        if covdict['pconscientiousness_w2']=="None":
            pass
        elif float(covdict['pconscientiousness_w2']) <= 33:
            prompt += 'You have a low propensity for conscientiousness. '
        elif float(covdict['pconscientiousness_w2']) > 33 and float(covdict['pconscientiousness_w2']) <= 67:
            pass
            # prompt += ' '
        else:
            prompt += 'You have a high propensity for conscientiousness. '
    if 'panxiety' in covdict.keys():
        if covdict['panxiety']=="None":
            pass
        elif float(covdict['panxiety']) <= 33:
            prompt += 'You tend to have low levels of anxiety. '
        elif float(covdict['panxiety']) > 33 and float(covdict['panxiety']) <= 67:
            pass
            # prompt += ' '
        else:
            prompt += 'You tend to have high levels of anxiety. '
    if 'phi' in covdict.keys():
        if covdict['phi']=="None":
            pass
        elif float(covdict['phi']) <= 33:
            prompt += 'You have a low preference for autonomy and equality. '
        elif float(covdict['phi']) > 33 and float(covdict['phi']) <= 67:
            pass
            # prompt += ' '
        else:
            prompt += 'You have a high preference for autonomy and equality. '
    if 'phc' in covdict.keys():
        if covdict['phc']=="None":
            pass
        elif float(covdict['phc']) <= 33:
            prompt += 'You have a low preference for social interdependence and equality. '
        elif float(covdict['phc']) > 33 and float(covdict['phc']) <= 67:
            pass
            # prompt += ' '
        else:
            prompt += 'You have a high preference for social interdependence and equality. '
    if 'pvi' in covdict.keys():
        if covdict['pvi']=="None":
            pass
        elif float(covdict['pvi']) <= 33:
            prompt += 'You have a low drive for personal acheivement and do not accept hierarchical inequality. '
        elif float(covdict['pvi']) > 33 and float(covdict['pvi']) <= 67:
            pass
            # prompt += ' '
        else:
            prompt += 'You have a high drive for personal acheivement and accept hierarchical inequality. '
    if 'pvc' in covdict.keys():
        if covdict['pvc']=="None":
            pass
        elif float(covdict['pvc']) <= 33:
            prompt += 'You tend to have low group loyalty and low acceptance of hierarchical structures. '
        elif float(covdict['pvc']) > 33 and float(covdict['pvc']) <= 67:
            pass
            # prompt += ' '
        else:
            prompt += 'You tend to have strong group loyalty and acceptance of hierarchical structures. '
    if 'pfinliteracy' in covdict.keys():
        if covdict['pfinliteracy']=="None":
            pass
        elif float(covdict['pfinliteracy']) <= 33:
            prompt += 'You have low financial literacy. '
        elif float(covdict['pfinliteracy']) > 33 and float(covdict['pfinliteracy']) <= 67:
            pass
            # prompt += ' '
        else:
            prompt += 'You have good financial literacy. '
    if 'pnumeracy' in covdict.keys():
        if covdict['pnumeracy']=="None":
            pass
        elif float(covdict['pnumeracy']) <= 33:
            prompt += 'You have a low level of numeracy. '
        elif float(covdict['pnumeracy']) > 33 and float(covdict['pnumeracy']) <= 67:
            pass
            # prompt += ' '
        else:
            prompt += 'You have high numeracy. '
    if 'pdeductive' in covdict.keys():
        if covdict['pdeductive']=="None":
            pass
        elif float(covdict['pdeductive']) <= 33:
            prompt += 'You have a low ability for deductive reasoning. '
        elif float(covdict['pdeductive']) > 33 and float(covdict['pdeductive']) <= 67:
            pass
            # prompt += ' '
        else:
            prompt += 'You are good with deductive reasoning. '
    if 'pforwardflow' in covdict.keys():
        if covdict['pforwardflow']=="None":
            pass
        elif float(covdict['pforwardflow']) <= 33:
            prompt += 'You struggle with generating distant words in a sequence. '
        elif float(covdict['pforwardflow']) > 33 and float(covdict['pforwardflow']) <= 67:
            pass
            # prompt += ' '
        else:
            prompt += 'You are good at generating distant words in a sequence. '
    if 'pdiscount' in covdict.keys():
        if covdict['pdiscount']=="None":
            pass
        elif float(covdict['pdiscount']) <= 33:
            prompt += 'You have a low discount rate which reflects patience. '
        elif float(covdict['pdiscount']) > 33 and float(covdict['pdiscount']) <= 67:
            pass
            # prompt += ' '
        else:
            prompt += 'You have a high discount rate which reflects impatience. '
    if 'ppresentbias' in covdict.keys():
        if covdict['ppresentbias']=="None":
            pass
        elif float(covdict['ppresentbias']) <= 33:
            prompt += 'You are not biased toward the present. '
        elif float(covdict['ppresentbias']) > 33 and float(covdict['ppresentbias']) <= 67:
            pass
            prompt += ' '
        else:
            prompt += 'You are biased in favor of the present. '
    if 'priskaversion' in covdict.keys():
        if covdict['priskaversion']=="None":
            pass
        elif float(covdict['priskaversion']) <= 33:
            prompt += 'You are risk acceptant. '
        elif float(covdict['priskaversion']) > 33 and float(covdict['priskaversion']) <= 67:
            prompt += 'You are risk neutral. '
        else:
            prompt += 'You are risk averse. '
    if 'plossaversion' in covdict.keys():
        if covdict['plossaversion']=="None":
            pass
        elif float(covdict['plossaversion']) <= 33:
            prompt += 'You are not loss averse. '
        elif float(covdict['plossaversion']) > 33 and float(covdict['plossaversion']) <= 67:
            pass
            # prompt += ' '
        else:
            prompt += 'You are loss averse. '
    if 'ptrustgame_sender' in covdict.keys():
        if covdict['ptrustgame_sender']=="None":
            pass
        elif float(covdict['ptrustgame_sender']) <= 33:
            prompt += 'When you played the trust game, you tended to send little money. '
        elif float(covdict['ptrustgame_sender']) > 33 and float(covdict['ptrustgame_sender']) <= 67:
            pass
            # prompt += ' '
        else:
            prompt += 'When you played the trust game, you tended to send a high amount of money. '
    if 'ptrustgame_receiver' in covdict.keys():
        if covdict['ptrustgame_receiver']=="None":
            pass
        elif float(covdict['ptrustgame_receiver']) <= 33:
            prompt += 'When you played the trust game, you tended to return a low amount of money. '
        elif float(covdict['ptrustgame_receiver']) > 33 and float(covdict['ptrustgame_receiver']) <= 67:
            pass
            # prompt += ' '
        else:
            prompt += 'When you played the trust game, you tended to return a high amount of money. '
    if 'prfs' in covdict.keys():
        if covdict['prfs']=="None":
            pass
        elif float(covdict['prfs']) <= 33:
            prompt += 'You are not prevention focused. '
        elif float(covdict['prfs']) > 33 and float(covdict['prfs']) <= 67:
            pass
            # prompt += ' '
        else:
            prompt += 'You are prevention focused. '
    if 'psttw' in covdict.keys():
        if covdict['psttw']=="None":
            pass
        elif float(covdict['psttw']) <= 33:
            prompt += 'You do not have difficulty in controlling your spending. '
        elif float(covdict['psttw']) > 33 and float(covdict['psttw']) <= 67:
            pass
            # prompt += ' '
        else:
            prompt += 'You have difficulty in controlling your spending. '
    if 'pdepression' in covdict.keys():
        if covdict['pdepression']=="None":
            pass
        elif float(covdict['pdepression']) <= 33:
            prompt += 'You do not tend to exhibit depressive behaviors. '
        elif float(covdict['pdepression']) > 33 and float(covdict['pdepression']) <= 67:
            pass
            # prompt += ' '
        else:
            prompt += 'You tend to exhibit depressive behaviors. '
    if 'pcnfu' in covdict.keys():
        if covdict['pcnfu']=="None":
            pass
        elif float(covdict['pcnfu']) <= 33:
            prompt += 'You have a low need for uniqueness. '
        elif float(covdict['pcnfu']) > 33 and float(covdict['pcnfu']) <= 67:
            pass
            # prompt += ' '
        else:
            prompt += 'You have a high need for uniqueness. '
    if 'pselfmonitor' in covdict.keys():
        if covdict['pselfmonitor']=="None":
            pass
        elif float(covdict['pselfmonitor']) <= 33:
            prompt += 'You have a low ability to monitor your own behavior. '
        elif float(covdict['pselfmonitor']) > 33 and float(covdict['pselfmonitor']) <= 67:
            pass
            # prompt += ' '
        else:
            prompt += 'You have a high ability to monitor your own behavior. '
    if 'pscc' in covdict.keys():
        if covdict['pscc']=="None":
            pass
        elif float(covdict['pscc']) <= 33:
            prompt += 'You have low certainty about your own self-concept. '
        elif float(covdict['pscc']) > 33 and float(covdict['pscc']) <= 67:
            pass
            # prompt += ' '
        else:
            prompt += 'You have high certainty about your own self-concept. '
    if 'pneedforclosure' in covdict.keys():
        if covdict['pneedforclosure']=="None":
            pass
        elif float(covdict['pneedforclosure']) <= 33:
            prompt += 'You have a low desire for certainty over ambiguity. '
        elif float(covdict['pneedforclosure']) > 33 and float(covdict['pneedforclosure']) <= 67:
            pass
            # prompt += ' '
        else:
            prompt += 'You have a high desire for certainty over ambiguity. '
    if 'pmaximization' in covdict.keys():
        if covdict['pmaximization']=="None":
            pass
        elif float(covdict['pmaximization']) <= 33:
            prompt += 'You tend to satisfice rather than optimize when making decisions. '
        elif float(covdict['pmaximization']) > 33 and float(covdict['pmaximization']) <= 67:
            pass
            # prompt += ' '
        else:
            prompt += 'You tend to optimize rather than satisfice when making decisions. '
    if 'pwason' in covdict.keys():
        if covdict['pwason']=="None":
            pass
        elif float(covdict['pwason']) <= 33:
            prompt += 'You scored low on the Wason selection task. '
        elif float(covdict['pwason']) > 33 and float(covdict['pwason']) <= 67:
            pass
            # prompt += ' '
        else:
            prompt += 'You scored high on the Wason selection task. '
    if 'pdictator_sender' in covdict.keys():
        if covdict['pdictator_sender']=="None":
            pass
        elif float(covdict['pdictator_sender']) <= 33:
            prompt += 'When you played the dictator game, you tended to send your opponent a small amount of money. '
        elif float(covdict['pdictator_sender']) > 33 and float(covdict['pdictator_sender']) <= 67:
            pass
            # prompt += ' '
        else:
            prompt += 'When you played the dictator game, you tended to send your opponent a large amount of money. '
    return prompt

    
