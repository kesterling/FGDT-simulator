# Focus Group Digital Twin Simulator

Built by the [TeCD-Lab](https://tecd-lab.ucr.edu/) at UCR, the focus group digitial twin (FGDT) simulator can evaluate the sampling distribution for different design implementations of the structured focus group. The structured focus group is an AI-assisted research design that can scale to any number of participants, and can be implemented for human participants using the online collaboration platform called [Prytaneum](https://prytaneum.io/). The FDGT simulator is a low cost, low effort method to evaluate the design of a structured focus group before conducting sessions with human participants.

## Introduction to the repository

This is the repository to run the focus group digital twins simulator developed by the TeCD-Lab at UCR. For questions about this repository, please write to Kevin Esterling, <kevin.esterling@ucr.edu>.

**To start, be certain to first read the paper entitled, "Using LLM Digital Twin Simulation to Evaluate the Emergent Properties of Human Group Interaction." That paper describes the software implemented in this repository.** The PDF is [available in the repository](https://github.com/kesterling/FGDT-simulator/blob/main/Using%20LLM%20Digital%20Twin%20Simulation%20to%20Evaluate%20the%20Emergent%20Properties%20of%20Human%20Group%20Interaction.pdf) 

## Instructions to run the simulator script

TO RUN THIS REPOSITORY, download the full zip file, unpack it, and run the script locally on your machine. The download will include four subdirectories that are necessary for the Python scripts to run. These are:

- MyPersonalKeyAPI (where you store your API keys)
- GooglegeminiCache (which caches the Gemini responses)
- GroqllmCache (which stores the Groq model caches)
- Results (which stores the simulation results for each run)

You must enter your Gemini API key as a single line in the file secretge.txt, and your Groq API key as a single line in the file secretgr.txt. Both of these files are located in the MyPersonalKeyAPI directory.


The Python script "simulation_digitaltwins_v2.py" runs the main simulator. To run the simulator, open the project directory in your IDE (such as VS Code or Positron), open the "simulation_digitialtwins_v2.py" file, edit the USER PARAMETERS in the main() function, and then press run. The parameters that you can set for the simulation are:

- *models:* a list of LLM models 
- *cov_keys:* a list of keys indicating covariates to use to build personas (keys are explained below)
- *topic:* a string of a few words defining the topic of the discussion
- *standpointOptions:* a list of closed-ended survey standpoint options related to the topic 
- *background:* a string providing the LLM any background information relevant to the topic
- *sampleSize:* an integer setting the sample size of LLM agents for each trial
- *numRuns:* an integer setting the number of trials per experiment; usually 50 small-N and 10-20 for large-N
- *saveToFile:* set to True to ave the results of each experiment run to file; keep as True unless testing
- *saveSimilaritiesToFile:* set to True to save the final cosine similarities to file
- *force* = False  # set to True if you don't want to use cached answers; caching allows you to re-run an experiment at no cost
 
The parameters are preset to values we use for the simulations we report in the paper, but these can be modified to fit the simulator's substantive interests. You do not need to know Python to run the script. The script gives examples and provide additional comments on how to set the paramters. When editing the user parameters, be certain to use the same formatting and structure as the preset values, ensuring lists are entered as lists, integers as integers, etc. The number of items you include in a list is arbitrary, however. For example, it does not matter how many standpointOptions you include or how many cov_keys you include. You can omit background information for the LLM by including '' (i.e., an empty string) as your string. You can omit covariates by only including 'model' in the cov_keys list. All other paramters must be set.

The digital twins data base comes from Toubia et al. (2025). See the file "DigitalTwin example.txt" to interpret the Toubia digital twin covariates. This file also provides the keys for each covariate that you can include to build a persona. The keys are entered in the cov_keys list in the USER PARAMETER section of the main() function. Importantly, the first key in the cov_keys list must be 'model' -- otherwise you can include any keys from the "DigitalTwin example.txt" file in any order. 

Each session will create a new folder in the Results subdirectory with name "trial_id_#" where the # is replaced by a random number. Each trial_id_# folder has a unique name so new experiements do not overwite previous experiments. When running an experimental session, the script will create a new trial_id_# folder and then write the following files into the new folder:

- a text file reporting the parameter settings used
- a set of CSV files, each with all personality covariates, options chosen, arguments seen, and responses. There is one CSV file per trial and the file name indicates the index of the trial
- a set of JSON-like files, each with the summaries across the three rounds, for each trial. Likewise, the file name indicates the index of the trial
- a CSV file for each option reporting the within-option cosine similarities across trials for that session

The files saved in each trial_id_# folder are the basic output of this simulator. 

## Validation scripts

There are a few auxiliary scripts available as well to run the validation tests. First, you can run "summarizationSimilarities.py" to update the CSV file to recover the cosine similarities from each stage 1 persona response and the corresponding summary for a given run. You use this updated file to do a summary bias evaluation as described in the paper. Second, you can use "coverage_test.py" to run the LLM coder for the coverage validation test that is also described in the paper. Optionally run "Round1_frequencies.py" to optimize the parameter settings in order to get the desired response frequencies across the standpointOptions. As the paper demonstrates, if these initial responses are skewed then you need more participants to adequately power a structured focus group, so this tool lets you check whehter the covariates you use to define personas are producing an optimal distribution of initial responses. You also can run "baseline_similarities.py" to infer the correlation between arbitrary sentences for a given topic, to get a random baseline for cosine similarity scores for that topic.

## A comment about replication versus reproduction

Running the simulator using the preset values will replicate, but not reproduce, the results we report in the paper. It is not possible to reproduce the simulation output since there is no way to set the seed for the LLM models we use. Further, Google has disabled the Gemini model we used gemini-2.0-flash. The scripts now send API calls to gemini-3-flash-preview. (The Groq models remain the same.) For these reasons, users cannot replicate the exact simulation results we report in the paper. However, since our paper focuses on the distribution of cosine similarity scores rather than point estimates for both the main results and the bias evaluation, the replicated results should be similar to the results we report in the paper. 

We provide the data and code to use our simulation output in order to reproduce the figures in our [reproduction epository](https://github.com/kesterling/FGDT-results). Our reproductioni repository shows we can reproduce the results reported in the paper from the raw data that resulted from our simulation runs. Reproduction in this context is analogous to analyzing survey data provided by a vendor. It would be strange to require a researcher to re-administer a survey and expect the survey results to be identical, even with the same respondents. 


## Repository file listing

The files in the respository are as follows:

- "simulation_digitalteins_v2.py" is the main simulator file
- "DigitalTwin example.txt" is the Toubia (2025) database codebook indicating covariates and their keys
- "GoogleGemini.py" and "GroqLlm.py" call the LLMs
- "digitaltwins_promptblocks.py" is the module containing the prompt blocks for the digital twin covariates
- "ProcessDigitalTwins.py" is the code that converts the Toubia (2025) digital twin data to a CSV file
- "twin_personalities_file.csv" is the digital twin covariate CSV file (i.e., the output of ProcessDigitalTwins.py)
- "summarizationSimilarities.py" is the tool used in the bias audit for the summaries validation
- "coverage_test.py" is the LLM coder to implement the coverage validation test
- "round1_frequencies.py" runs round1 only, so just outputs normal survey responses
- "baseline_similarities.py" creates random sentences on a topic to establish a baseline cosine similarity distribution

## References

Toubia, O., G. Z. Gui, T. Peng, D. J. Merlau, A. Li, and H. Chen (2025, 8). Database Report: Twin-2K-500: A Data Set for Building Digital Twins of over 2,000 People Based on Their Answers to over 500 Questions. _Marketing Science_ 44 (6), 1446–1455.
