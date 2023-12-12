import os
import openai
import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import pairwise_distances
import random
import nltk
from nltk.corpus import wordnet
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
import Levenshtein
openai_api_key = "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"

temperature_values = [0.2, 0.5, 0.7, 1.0]
max_tokens_values = 150

def load_problems(directory):
    problems_data = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith('.json'):
            with open(os.path.join(directory, filename), 'r') as json_file:
                problems = json.load(json_file)
                for problem in problems['prompts']:
                    problems_data.append((problem['base'], filename))
    return problems_data

# Function to get LLM responses with a maximum character limit
def get_llm_response(prompt, openai_api_key, temperature=0.7, max_tokens=150):
    openai.api_key = openai_api_key
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        content = response['choices'][0]['message']['content'].strip()
        description, code = extract_code(content)
        return {'description': description, 'code': code}
    except openai.error.OpenAIError as e:
        print(f"An error occurred: {e}")
        return {'description': None, 'code': None}



def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        synonyms.update(lemma.name().replace('_', ' ') for lemma in syn.lemmas())
    return list(synonyms)

def replace_synonyms(sentence):
    words = nltk.word_tokenize(sentence)
    pos_tags = nltk.pos_tag(words)
    candidates = [(word, i) for i, (word, pos) in enumerate(pos_tags) if pos in ["NN", "JJ"]]
    if candidates:
        to_replace, idx = random.choice(candidates)
        synonyms = get_synonyms(to_replace)
        if synonyms:
            sentence = words[:idx] + [random.choice(synonyms)] + words[idx+1:]
            return ' '.join(sentence)
    return sentence

def generate_variations(base_prompt):
    variation1 = replace_synonyms(base_prompt)
    if base_prompt.startswith("What is the"):
        variation2 = base_prompt.replace("What is the", "Can you tell me what the", 1)
    else:
        variation2 = "Can you tell me: " + base_prompt
    variation3 = base_prompt.replace("What is the", "I'm curious about the", 1)
    return [variation1, variation2, variation3]

def extract_code(content):
    # Split the content by triple backticks
    parts = content.split('```')
    
    # Initialize variables for description and code
    description = ""
    code = None

    # Process each part
    for i in range(0, len(parts), 2):
        if i < len(parts):
            description += parts[i]
        if i + 1 < len(parts):
            code = parts[i + 1].strip()  # Get the code block

    return description.strip(), code




# Function to calculate Jaccard Similarity
def calculate_jaccard_similarity(response1, response2):
    vectorizer = CountVectorizer(binary=True)
    vectors = vectorizer.fit_transform([response1, response2]).toarray()
    return 1 - pairwise_distances(vectors[0].reshape(1, -1), vectors[1].reshape(1, -1), metric = 'jaccard')[0][0]

def calculate_levenshtein_distance(str1, str2):
    return Levenshtein.distance(str1, str2)

def analyze_prompts(problems_data, openai_api_key):
    responses_directory = "responses"
    scores_directory = "scores"
    os.makedirs(responses_directory, exist_ok=True)
    os.makedirs(scores_directory, exist_ok=True)

    for base_prompt, filename in problems_data:
        problem_number = filename.replace('.json', '')

        base_response_data = get_llm_response(base_prompt, openai_api_key)
        if base_response_data['description'] is not None:  # Check if a response was received
            base_code = base_response_data['code']  # Get the code part of the base response
            variations = generate_variations(base_prompt)
            variation_responses = {}

            for var in variations:
                var_response_data = get_llm_response(var, openai_api_key)
                if var_response_data['description'] is not None:
                    var_code = var_response_data['code']  # Get the code part of the variant response
                    variation_responses[var] = {
                        'description': var_response_data['description'],
                        'code': var_code
                    }

            # Calculate and save scores
            similarities = []
            distances = []

            for var, var_response in variation_responses.items():
                if base_code and var_response['code']:
                    sim = calculate_jaccard_similarity(base_code, var_response['code'])
                    dist = calculate_levenshtein_distance(base_code, var_response['code'])
                    similarities.append(sim)
                    distances.append(dist)

            # Save responses to JSON file in the 'responses' folder
            responses_file_path = os.path.join(responses_directory, f"{problem_number}.json")
            with open(responses_file_path, 'w') as json_file:
                json.dump({
                    'base_prompt': base_prompt,
                    'base_response': base_response_data['description'],
                    'base_code': base_code,
                    'variation_responses': variation_responses
                }, json_file, indent=4)

            # Save scores to JSON file in the 'scores' folder
            scores_file_path = os.path.join(scores_directory, f"{problem_number}.json")
            with open(scores_file_path, 'w') as json_file:
                json.dump({
                    'prompt': base_prompt,
                    'similarities': similarities,
                    'mean_similarity': np.mean(similarities) if similarities else None,
                    'distances': distances,
                    'mean_distance': np.mean(distances) if distances else None
                }, json_file, indent=4)

directory_path = "prompts"
responses_directory = "responses"
scores_directory = "scores"
os.makedirs(responses_directory, exist_ok=True)
os.makedirs(scores_directory, exist_ok=True)

# Load the prompts from the directory
loaded_prompts = load_problems(directory_path)

# Perform experiments with different parameters
for temperature in temperature_values:
        print(f"Running experiments with Temperature={temperature}")
        
        for base_prompt, filename in loaded_prompts:
            problem_number = filename.replace('.json', '')

            base_response_data = get_llm_response(base_prompt, openai_api_key, temperature=temperature)
            
            if base_response_data['description'] is not None:
                base_code = base_response_data['code']
                variations = generate_variations(base_prompt)
                variation_responses = {}

                for var in variations:
                    var_response_data = get_llm_response(var, openai_api_key, temperature=temperature)
                    
                    if var_response_data['description'] is not None:
                        var_code = var_response_data['code']
                        variation_responses[var] = {
                            'description': var_response_data['description'],
                            'code': var_code
                        }

                # Calculate and save scores
                similarities = []
                distances = []

                for var, var_response in variation_responses.items():
                    if base_code and var_response['code']:
                        sim = calculate_jaccard_similarity(base_code, var_response['code'])
                        dist = calculate_levenshtein_distance(base_code, var_response['code'])
                        similarities.append(sim)
                        distances.append(dist)

                # Save responses to JSON file in the 'responses' folder
                responses_file_path = os.path.join(responses_directory, f"{problem_number}_T{temperature}.json")
                with open(responses_file_path, 'w') as json_file:
                    json.dump({
                        'base_prompt': base_prompt,
                        'base_response': base_response_data['description'],
                        'base_code': base_code,
                        'variation_responses': variation_responses
                    }, json_file, indent=4)

                # Save scores to JSON file in the 'scores' folder
                scores_file_path = os.path.join(scores_directory, f"{problem_number}_T{temperature}.json")
                with open(scores_file_path, 'w') as json_file:
                    json.dump({
                        'prompt': base_prompt,
                        'similarities': similarities,
                        'mean_similarity': np.mean(similarities) if similarities else None,
                        'distances': distances,
                        'mean_distance': np.mean(distances) if distances else None
                    }, json_file, indent=4)

# directory_path = "prompts"
# loaded_prompts = load_problems(directory_path)



# analyze_prompts(loaded_prompts, openai_api_key)
