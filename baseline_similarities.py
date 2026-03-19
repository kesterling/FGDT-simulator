import os
from google import genai 
from google.genai import types
# import hashlib
import ast
from sentence_transformers import SentenceTransformer # Assigns numerical values to text to help with paraphrasing/finding meaning within a sentence (text -> vector)
from sklearn.metrics.pairwise import cosine_similarity # Uses the numerical values assigned to compare them  and guage if sentences have the same meaning!!
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

folder = os.path.dirname(os.path.abspath(__file__))  # Folder of this script
with open(folder + "/MyPersonalKeyAPI/secretge.txt", 'r', encoding="utf-8") as f:
    apikey = f.read() # Personal key
os.environ['GEMINI_API_KEY'] = apikey # Personal key
# The next two lines create a client instance
global client 
client = genai.Client()
model='gemini-3-flash-preview'

prompt="Please write 100 sentences about nuclear power in the US. Please return the sentences as a Python list."

completion = client.models.generate_content(
    model=model,
    contents=prompt,
    config=types.GenerateContentConfig(
        # max_output_tokens= max_output_tokens,
        top_k= 40,
        temperature= 0.5,  # Randomness: Low temp = low randomness, high temp = high creativity
        safety_settings= [
            types.SafetySetting(
                category='HARM_CATEGORY_HATE_SPEECH',
                threshold='BLOCK_NONE'
                # Add other safety categories and thresholds here....
                ),
            ]
        ),
    )


response = completion.text
startIndex = response.find('[') 
endIndex = response.find(']')+1
response = response[startIndex:endIndex]
response = response.replace('\n', '')
sentences=ast.literal_eval(response)  # converts the string to a list

transformerModel = SentenceTransformer('all-MiniLM-L6-v2')

sentenceEmbeddings = transformerModel.encode(sentences)
similarityMatrix = cosine_similarity(sentenceEmbeddings)

upperTriangle = np.triu_indices_from(similarityMatrix, k=1) # Get the upper triangle indices of the matrix, excluding the diagonal
similaritymean = np.mean(similarityMatrix[upperTriangle])
print(similaritymean)
similaritysd = np.std(similarityMatrix[upperTriangle])
print(similaritysd)


pd.DataFrame(similarityMatrix[upperTriangle]).plot.density()

rng = np.random.default_rng()
y=str(int(rng.integers(10**10)))
np.savetxt(f'Results/similarityMatrix_randombaseline_n{y}.csv', similarityMatrix,
                           delimiter=',', fmt='%.2f')

