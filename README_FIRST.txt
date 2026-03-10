
*** DO NOT RUN THIS FILE IN THE OFFICIAL REPOSITORY! 
	You must first create your own clone of the repository and then run it there.
	This keeps the cache in the official repository clean and prevents overwriting of any files


For questions about this directory, please write to Kevin Esterling, kevin.esterling@ucr.edu


See "DigitalTwin example.txt" to interpret the digital twin covariates
Optionally run "Round1_frequencies.py" to optimize the parameter settings in order to get the desired response frequencies 
Run "simulation_digitaltwins_v2.py" with your optimized parameter settings. Each session will create a new directory with
- a text file reporting the parameter settings used
- a CSV file with all personality covariates, options chosen, arguments seen, and responses
- a JSON-like file with the summaries across the three rounds
- a CSV file for each option reporting the within-option cosine similarities
Then run "summarizationSimilarities.py" to update the CSV file to recover the cosine similarities from each round 1 response and the corresponding summary for a given run. You use this updated file to do a bias evaluation.
You also can run "baseline_similarities.py" to infer the correlation between arbitrary sentences for a given topic.

"GoogleGemini.py" and "GroqLlm.py" call the LLMs
"digitaltwins_promptblocks.py" is the module containing the prompt blocks for the digital twin covariates
"ProcessDigitalTwins.py" is the code that converts the digital twin summaries to a CSV file
"twin_personalities_file.csv" is the digital twin covariate CSV file
"Ideas.txt" is for brainstorming analyses
"temp.py" is a scratch file for testing blocks of code -- please ignore.
"simulation_withcomments_v1.py" is the original (version 1.0) version of the simulator, written by Ben Treves, that does not use digital twin data to create personalities

