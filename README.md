This is the repository to run the focus group digital twins simulator developed by the TeCD-Lab at UCR. For questions about this repository, please write to Kevin Esterling, kevin.esterling@ucr.edu.

**To start, be certain to first read the paper (PDF is available in the repository) entitled, "Using LLM Digital Twin Simulation to Evaluate the Emergent Properties of Human Group Interaction." That paper describes the software implemented in this repository.**

TO RUN THIS REPOSITORY, download the full zip file, unpack it, and run the script locally on your machine. The download will include four subdirectories that are necessary for the Python scripts to run. These are:

- MyPersonalKeyAPI (where you store your API keys)
- GooglegeminiCache (which caches the Gemini responses)
- GroqllmCache (which stores the Groq model caches)
- Results (which stores the simulation results for each run)

You must enter your Gemini API key as a single line in the file secretge.txt, and your Groq API key as a single line in the file secretgr.txt. Both of these files are located in the MyPersonalKeyAPI directory.


The Python script "simulation_digitaltwins_v2.py" runs the main simulator. To run the simulator, edit the USER PARAMETERS in the main() function, and then press run. The parameters that you can set for the simulation are:

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
 
The script gives examples and provide additional comments on how to set the paramters. See "DigitalTwin example.txt" to interpret the digital twin covariates. This file also provides the keys for each covariate that you can include to build a persona. The keys are entered in the cov_keys list in the USER PARAMETER section of the main() function. Importantly, the first key in the cov_keys list must be 'model' -- otherwise you can include any keys from the "DigitalTwin example.txt" file in any order. 

Each session will create a new folder in the Results subdirectory with name "trial_id_#" where the # is replaced by a random number. Each trial_id_# folder has a unique name so new experiements do not overwite previous experiments. When running an experimental session, the script will create a new trial_id_# folder and then write the following files into the new folder:

- a text file reporting the parameter settings used
- a CSV file with all personality covariates, options chosen, arguments seen, and responses
- a JSON-like file with the summaries across the three rounds
- a CSV file for each option reporting the within-option cosine similarities

The files saved in each trial_id_# folder are the basic output of this simulator. There are a few auxiliary scripts available as well to run the validation tests. First, you can run "summarizationSimilarities.py" to update the CSV file to recover the cosine similarities from each round 1 response and the corresponding summary for a given run. You use this updated file to do a summary bias evaluation. Second, you can use "coverage_test.py" to run the LLM coder for the coverage validation test. Optionally run "Round1_frequencies.py" to optimize the parameter settings in order to get the desired response frequencies. You also can run "baseline_similarities.py" to infer the correlation between arbitrary sentences for a given topic, to get a random baseline for cosine similarity scores for that topic.


Repository file listing:
- "simulation_digitalteins_v2.py" is the main simulator file
- "DigitalTwin example.txt" is the codebook indicating covariates and their keys
- "GoogleGemini.py" and "GroqLlm.py" call the LLMs
- "digitaltwins_promptblocks.py" is the module containing the prompt blocks for the digital twin covariates
- "ProcessDigitalTwins.py" is the code that converts the digital twin summaries to a CSV file
- "twin_personalities_file.csv" is the digital twin covariate CSV file
- "baseline_similarities.py" creates random sentences on a topic to establish a baseline cosine similarity distribution
- "round1_frequencies.py" runs round1 only, so just outputs normal survey responses
- "summarizationSimilarities.py" is the tool used in the bias audit for the summaries
- "coverage_test.py" is the LLM coder to implement the coverage validation test
