import os
import json
import numpy as np

def calculate_average_scores(scores_directory):
    similarity_scores = []
    distance_scores = []

    # Loop over the files in the scores directory
    for filename in os.listdir(scores_directory):
        if filename.endswith('.json'):
            with open(os.path.join(scores_directory, filename), 'r') as json_file:
                data = json.load(json_file)
                if 'mean_similarity' in data and 'mean_distance' in data:
                    similarity_scores.append(data['mean_similarity'])
                    distance_scores.append(data['mean_distance'])

    # Calculate the average scores
    average_similarity = np.mean(similarity_scores) if similarity_scores else None
    average_distance = np.mean(distance_scores) if distance_scores else None

    return average_similarity, average_distance

def write_output(average_similarity, average_distance, output_file):
    # Scale the average similarity to a score out of 100
    similarity_score = average_similarity * 100 if average_similarity is not None else None

    # Normalize and scale the average distance to a score out of 100, assuming max distance of a reasonable value
    max_reasonable_distance = 20  # This is an example value; adjust based on what you deem reasonable
    distance_score = (1 - min(average_distance / max_reasonable_distance, 1)) * 100 if average_distance is not None else None

    # Combine the two scores for a final output score, with more weight given to similarity
    if similarity_score is not None and distance_score is not None:
        overall_score = (similarity_score * 0.75 + distance_score * 0.25)
    else:
        overall_score = None  # If either score is None, we cannot calculate an overall score

    # Prepare the data to be written as JSON
    output_data = {
        'output': overall_score,
        'average_similarity': average_similarity,
        'average_distance': average_distance
        
    }
    
    # Write the average scores to output.json
    with open(output_file, 'w') as json_out:
        json.dump(output_data, json_out, indent=4)

# Specify the directory where the score files are located
scores_directory = 'scores'

# Calculate the average scores
avg_similarity, avg_distance = calculate_average_scores(scores_directory)

# Specify the output file name
output_file = 'output.json'

# Write the output to the file
write_output(avg_similarity, avg_distance, output_file)

print(f"Average similarity and distance scores written to {output_file}")
