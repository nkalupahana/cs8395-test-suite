import os
import json
import matplotlib.pyplot as plt

# Path to the directory containing the score files
scores_directory = 'scores'
mean_distances = []

# Iterate over each file in the scores directory
for filename in os.listdir(scores_directory):
    if filename.endswith('.json'):
        # Construct the full file path
        file_path = os.path.join(scores_directory, filename)
        # Open and read the JSON file
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
            # Extract the mean distance and add it to the list
            if 'mean_distance' in data:
                mean_distances.append(data['mean_distance'])

# Create a boxplot of the mean distances
plt.figure(figsize=(10, 6))
plt.boxplot(mean_distances, vert=True, patch_artist=True) # Vert sets the boxplot orientation to vertical
plt.title('Levenshtein Distance Boxplot')
plt.ylabel('Mean Levenshtein Distance')

# Add a custom x-axis label if necessary
plt.xticks([1], ['Mean Levenshtein Distance'])

image_path = 'leven_boxplot.png'
plt.savefig(image_path, bbox_inches='tight')

# Show the plot
plt.show()
