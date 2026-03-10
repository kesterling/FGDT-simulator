import os
from google import genai 
from google.genai import types
# https://ai.google.dev/gemini-api/docs/migrate
# from google.generativeai.types import HarmCategory, HarmBlockThreshold
import hashlib



def InitGoogleGemini(folder=''):
    "Retrieve my API key and initialize Gemini with it"
    folder = os.path.dirname(os.path.abspath(__file__))  # Folder of this script
    # print(folder)
    with open(folder + "/MyPersonalKeyAPI/secretge.txt", 'r', encoding="utf-8") as f:
        apikey = f.read() # Personal key
    os.environ['GEMINI_API_KEY'] = apikey # Personal key
    # The next two lines create a client instance
    global client 
    client = genai.Client()



def AskGoogleGemini(prompt: str, model='gemini-2.5-flash', max_output_tokens=1024, force=False, temperature=0.5, top_k=40) -> str:
    "Ask a prompt to given Google Cloud model and return the response text and safety ratings. NOTE: Must run InitGoogleGemini() beforehand."
    # Get cache folder path of desired model and create one if it does not already exist
    folder = os.path.dirname(os.path.abspath(__file__)) + '/' # Folder of this script
    cachefolder = folder + 'GooglegeminiCache/' + model.replace('.', '_') + '/' # Folder names cannot have periods
    os.makedirs(cachefolder, exist_ok=True)


    # # Construct non-limiting safety filters 
    # --> the commented safety settings are from the deprecated generativeai SDK; keeping here for now...
    # safety_settings = []
    # for category in HarmCategory:
    #     safety_settings.append({
    #         "category": category,
    #         "threshold": HarmBlockThreshold.BLOCK_NONE,
    #     })
    
    # safety_settings=[
    #         {
    #             "category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
    #             "threshold": HarmBlockThreshold.BLOCK_NONE,
    #         },
    #         {
    #             "category": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
    #             "threshold": HarmBlockThreshold.BLOCK_NONE,
    #         },
    #         {
    #             "category": HarmCategory.HARM_CATEGORY_HARASSMENT,
    #             "threshold": HarmBlockThreshold.BLOCK_NONE,
    #         },
    #         {
    #             "category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
    #             "threshold": HarmBlockThreshold.BLOCK_NONE,
    #         },
    #     ]

    
    # Check if the prompt has been executed before
    response = ''
    hashedPrompt = str(hashlib.md5(prompt.encode('utf-8')).hexdigest()[:8])
    filepath = cachefolder + hashedPrompt
    if(os.path.isfile(filepath)):
        with open(filepath, 'r', encoding="utf-8") as f:
            response = f.read()

    # If force is True, always get a new response from Gemini
    if(force):
        response = ''

    # Get the response and its safety ratings from Gemini if it was not cached
    if(response == ''):
        completion = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            # max_output_tokens= max_output_tokens,
            top_k= top_k,
            temperature= temperature,  # Randomness: Low temp = low randomness, high temp = high creativity
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
        if(response is None):
            response = 'unknown'
        
        # Output the response and its safety ratings to cache if it has not been executed before
        with open(filepath, 'w', encoding="utf-8") as f:
            f.write(response)
    # print(f'Gemini resp {response}')
    return response