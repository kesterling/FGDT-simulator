from groq import Groq
import os
import hashlib

def InitGroq():
    "Retrieve my API key and initialize Groq with it"
    folder = os.path.dirname(os.path.abspath(__file__)) + '/' # Folder of this script
    with open(folder + 'MyPersonalKeyAPI/secretgr.txt', 'r', encoding="utf-8") as f:
        apikey = f.read() # Personal key
    os.environ['GROQ_API_KEY'] = apikey


def AskGroq(prompt: str, model='llama-3.3-70b-versatile', max_output_tokens=1024, force=False, temperature=0.5) -> str:
    """Ask a prompt to the given LLM model and return the result.
    All available models found at: https://console.groq.com/docs/models.
    NOTE: Must run InitGroq() beforehand.
    Ex model: llama-3.3-70b-versatile
    Ex model: gemma2-9b-it"""
    # Get cache folder path of desired model and create one if it does not already exist
    folder = os.path.dirname(os.path.abspath(__file__)) + '/' # Folder of this script
    cachefolder = folder + 'GroqllmCache/' + model.replace('.', '_') + '/' # Folder names cannot have periods
    os.makedirs(cachefolder, exist_ok=True)
    
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

    # Get the response from Groq if it was not cached
    if(response == ''):
        client = Groq()
        completion = client.chat.completions.create(
            model=model,
            messages=[{
                'role': 'user',
                'content': prompt,
            }],
            max_completion_tokens=max_output_tokens,
            temperature=temperature,
        )
        response = completion.choices[0].message.content
        
        # Output the response to cache if it has not been executed before
        with open(filepath, 'w', encoding="utf-8") as f:
            f.write(response)
   # print(f'Groq resp {response}')
    return response

def main():
    "Test the functionality of Groq LLMs"
    # Create a prompt to ask a Groq LLM
    model = 'gemma2-9b-it'
    prompt = 'What is the difference between a circle and a sphere?'

    # Initialize Groq
    InitGroq()

    # Get and print the response
    response = AskGroq(prompt, model)
    print(response)

if __name__ == '__main__':
    main()