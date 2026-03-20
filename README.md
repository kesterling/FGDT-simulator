This is the repository to run the focus group digital twins simulator developed by the TeCD-Lab at UCR. For questions about this repository, please write to Kevin Esterling, kevin.esterling@ucr.edu.

**To start, be certain to first read the paper (PDF is available in the repository) entitled, "Using LLM Digital Twin Simulation to Evaluate the Emergent Properties of Human Group Interaction." That paper describes the software implemented in this repository.**

TO RUN THIS REPOSITORY, download the full zip file, unpack it, and run the script locally on your machine. The download will include four subdirectories that are necessary for the Python scripts to run. These are:

- MyPersonalKeyAPI (where you store your API keys)
- GooglegeminiCache (which caches the Gemini responses)
- GroqllmCache (which stores the Groq model caches)
- Results (which stores the simulation results for each run)

You must enter your Gemini API key as a single line in the file secretge.txt, and your Groq API key as a single line in the file secretgr.txt. Both of these files are located in the MyPersonalKeyAPI directory.


The Python script "simulation_digitaltwins_v2.py" runs the main simulator. To run the simulator, edit the USER PARAMETERS in the main() function, and then press run. See "DigitalTwin example.txt" to interpret the digital twin covariates. This file also provides the keys for each covariate that you can include to build a persona. The keys are entered in the USER PARAMETER section of the main() function. Each session will create a new directory with:

- a text file reporting the parameter settings used
- a CSV file with all personality covariates, options chosen, arguments seen, and responses
- a JSON-like file with the summaries across the three rounds
- a CSV file for each option reporting the within-option cosine similarities

You can run "summarizationSimilarities.py" to update the CSV file to recover the cosine similarities from each round 1 response and the corresponding summary for a given run. You use this updated file to do a summary bias evaluation. Use "coverage_test.py" to run the LLM coder for the coverage validation test. Optionally run "Round1_frequencies.py" to optimize the parameter settings in order to get the desired response frequencies. You also can run "baseline_similarities.py" to infer the correlation between arbitrary sentences for a given topic, to get a random baseline for cosine similarity scores for that topic.


File listing:
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
