from secrets_openai import SECRET_KEY_OPENAI
nkita
import pandas as pd
nkita
import argparse
nkita
from tqdm import tqdm
nkita
import requests
nkita
import openai
nkita
import os
nkita
import json
nkita
import time
nkita
import icecream as ic
nkita

nkita
openai_api_key = SECRET_KEY_OPENAI
nkita
openai.api_key = openai_api_key
nkita

nkita
# Reference for library error type and method to handle
nkita
# https://platform.openai.com/docs/guides/error-codes/python-library-error-types
nkita

nkita
def create_chat_completion(model, system_message, user_prompt, val_file, temperature, max_tokens):
nkita
    try:
nkita
        response = openai.ChatCompletion.create(
nkita
            model=model,
nkita
            messages=[
nkita
        {"role": "system", "content": system_message},
nkita
        {"role": "system", "name":"example_user", "content": str(val_file['sent'][0])},
nkita
        {"role": "system", "name": "example_assistant", "content": str(val_file['facts'][0])},
nkita
        {"role": "system", "name":"example_user", "content": str(val_file['sent'][1])},
nkita
        {"role": "system", "name": "example_assistant", "content": str(val_file['facts'][1])},
nkita
        {"role": "system", "name":"example_user", "content": str(val_file['sent'][2])},
nkita
        {"role": "system", "name": "example_assistant", "content": str(val_file['facts'][2])},
nkita
        {"role": "system", "name":"example_user", "content": str(val_file['sent'][3])},
nkita
        {"role": "system", "name": "example_assistant", "content": str(val_file['facts'][3])},
nkita
        {"role": "system", "name":"example_user", "content": str(val_file['sent'][4])},
nkita
        {"role": "system", "name": "example_assistant", "content": str(val_file['facts'][4])},
nkita
        {"role": "user", "content": user_prompt}
nkita
            ],
nkita
            temperature=temperature,
nkita
            max_tokens=max_tokens,
nkita
        ) 
nkita
        return response
nkita
    except openai.error.APIError as e:
nkita
        print(f"OpenAI API returned an API Error: {e}")
nkita
        time.sleep(10)
nkita
        return create_chat_completion(model, system_message, user_prompt, temperature, max_tokens)
nkita
    except openai.error.APIConnectionError as e:
nkita
        print(f"Failed to connect to OpenAI API: {e}")
nkita
        time.sleep(10)
nkita
        return create_chat_completion(model, system_message, user_prompt, temperature, max_tokens)
nkita
    except openai.error.RateLimitError as e:
nkita
        print(f"OpenAI API request exceeded rate limit: {e}")
nkita
        # Exponential backoff logic can be added here.
nkita
        # But for now, we will just wait for 60 seconds before retrying as is based on TPM.
nkita
        # https://platform.openai.com/docs/guides/rate-limits/what-are-the-rate-limits-for-our-api
nkita
        time.sleep(60)
nkita
        return create_chat_completion(model, system_message, user_prompt, temperature, max_tokens)
nkita
    except openai.error.AuthenticationError as e:
nkita
        print(f"Authentication error with OpenAI API: {e}")
nkita
        # Handle authentication error here. You may need to regenerate your API key.
nkita
        return None
nkita
    except openai.error.InvalidRequestError as e:
nkita
        print(f"Invalid request error with OpenAI API: {e}")
nkita
        # Handle invalid request error here. You may need to check your request parameters.
nkita
        return None
nkita
    except openai.error.ServiceUnavailableError as e:
nkita
        print(f"Service unavailable error with OpenAI API: {e}")
nkita
        time.sleep(30)
nkita
        return create_chat_completion(model, system_message, user_prompt, temperature, max_tokens)
nkita
    except requests.exceptions.ReadTimeout as e:
nkita
        print(f"Network request took too long: {e}")
nkita
        time.sleep(60)  # Wait for 60 seconds before retrying
nkita
        return create_chat_completion(model, system_message, user_prompt, temperature, max_tokens)
nkita
    except Exception as e:
nkita
        print(f"An unexpected error occurred: {e}")
nkita
        time.sleep(600)
nkita
        return create_chat_completion(model, system_message, user_prompt, temperature, max_tokens)
nkita

nkita

nkita
def chat_api(prompt_context, LANG, val_file, system_message="", temperature=0):
nkita
    system_message = f"You must extract all facts in English from the following {LANG} sentence. A fact consists of a relation and tail entity present in the sentence. Return the extracted facts in the form of a list of lists."
nkita

nkita
    # Read from prompt-esconv-strategy.txt as text
nkita

nkita
    prompt = prompt_context
nkita
    # Ref: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb
nkita
    # Should work with "gpt-4" if you have access to it.
nkita
    ANNOTATING_MODEL = "gpt-4"
nkita
    # response = openai.ChatCompletion.create(
nkita
    #     model=ANNOTATING_MODEL,
nkita
    #     messages=[
nkita
    #         {"role": "system", "content": system_message},
nkita
    #         {"role": "user", "content": prompt}
nkita
    #     ],
nkita
    #     temperature=temperature,
nkita
    #     max_tokens=900,
nkita
    # )
nkita
    # print(prompt)
nkita
    response = create_chat_completion(ANNOTATING_MODEL, system_message, prompt, val_file, temperature, 1000)
nkita
    response_content = response['choices'][0]['message']['content']
nkita
    return response_content
nkita

nkita
if __name__ == "__main__":
nkita
    parser = argparse.ArgumentParser(description='Input parameters for gpt-4')
nkita
    parser.add_argument('--language', help='language of the sentences')
nkita
    parser.add_argument('--val_csv', help='path to the validation csv files')
nkita
    parser.add_argument('--test_csv', help='path to the test csv files')
nkita
    parser.add_argument('--output_csv', help='path to store the output csv containing the OpenAI results')
nkita
    
nkita
    args = parser.parse_args()
nkita
    
nkita
    languages_map = {
nkita
    'bn': {"label": "Bengali"},
nkita
    'en': {"label": "English"},
nkita
    'hi': {"label": "Hindi"},
nkita
    'or': {"label": "Odia"},
nkita
    'pa': {"label": "Punjabi"},
nkita
    'ta': {"label": "Tamil"},    
nkita
}
nkita
    lang_code = args.language
nkita
    LANG = languages_map[lang_code]['label']
nkita
        
nkita
    val_csv_path = args.val_csv
nkita
    test_csv_path = args.test_csv
nkita
    val_file = pd.read_csv(val_csv_path)
nkita
    test_file = pd.read_csv(test_csv_path)
nkita
    
nkita
    output_csv_path = args.output_csv
nkita

nkita
    val_file = val_file[0:5]
nkita
    #change
nkita
    #test_file = test_file[0:5]
nkita
    
nkita
    output = []
nkita

nkita
    
nkita
    for i in tqdm(range(len(test_file))):
nkita
        response_content = chat_api(str(test_file['sent'][i]),LANG, val_file)
nkita

nkita
        output.append({
nkita
                'sent': str(test_file['sent'][i]),
nkita
                'facts': str(test_file['facts'][i]),
nkita
                'pred_facts': response_content,
nkita
            })
nkita
        current_df = pd.DataFrame([output[-1]])
nkita
        current_df.to_csv(output_csv_path, mode='a', header=not i, index=False)
nkita
        
nkita
    print("Done!")
nkita
