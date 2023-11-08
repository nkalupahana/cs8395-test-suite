import matplotlib.pyplot as plt
import json
import os
import numpy as np

# Replace 'scores_directory' with your actual scores directory path
scores_directory = 'scores'
files = sorted([f for f in os.listdir(scores_directory) if f.endswith('.json')])

# Initialize lists to store the mean distances and problem numbers
mean_levenshtein_distances = []
problem_numbers = []

# Read each JSON file and extract the mean Levenshtein distance
for file in files:
    with open(os.path.join(scores_directory, file), 'r') as json_file:
        data = json.load(json_file)
        if data['distances']:  # Check if there are distances recorded
            mean_distance = np.mean(data['distances'])
            mean_levenshtein_distances.append(mean_distance)
            # Assuming the file name format "problem[number].json"
            problem_number = int(file.replace("problem", "").replace(".json", ""))
            problem_numbers.append(problem_number)

# Sort the problem numbers and their corresponding distances
sorted_indices = np.argsort(problem_numbers)
sorted_problem_numbers = np.array(problem_numbers)[sorted_indices]
sorted_distances = np.array(mean_levenshtein_distances)[sorted_indices]

# Create a line chart
plt.figure(figsize=(10, 5))
plt.plot(sorted_problem_numbers, sorted_distances, marker='o', linestyle='-', color='b')

# Adding labels and title
plt.xlabel('Problems')
plt.ylabel('Mean Levenshtein Distance')
plt.title('Mean Levenshtein Distance for Each Problem')

# Save the figure
output_image_path = 'mean_levenshtein_distance_chart.png'
plt.savefig(output_image_path, bbox_inches='tight')

# Display the plot
plt.show()

print(f"The mean Levenshtein distance chart has been saved as an image here: {output_image_path}")
