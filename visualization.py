import os
import json
import matplotlib.pyplot as plt

# Path to the directory containing the score files
scores_directory = 'scores'
all_distances = []

# Problem labels for x-axis
problem_labels = []

# Iterate over each file in the scores directory
for filename in os.listdir(scores_directory):
    if filename.endswith('.json'):
        # Construct the full file path
        file_path = os.path.join(scores_directory, filename)
        # Open and read the JSON file
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
            # Extract the distances for each problem and add them to the list
            if 'distances' in data:  # Assuming 'distances' is a list of all distances for a problem
                all_distances.append(data['distances'])
                problem_labels.append(filename[:-5])  # Remove '.json' from the filename for the label

# Check if we have data to plot
if not all_distances:
    print("No data to plot.")
else:
    # Create a boxplot of the distances for all problems
    plt.figure(figsize=(10, 6))
    plt.boxplot(all_distances, vert=True, patch_artist=True) # Vert sets the boxplot orientation to vertical
    plt.title('Levenshtein Distance Boxplot for All Problems')
    plt.ylabel('Levenshtein Distance')

    # Add custom x-axis labels based on the problem names
    plt.xticks(range(1, len(problem_labels)+1), problem_labels, rotation=45, ha='right')

    image_path = 'leven_boxplot_all_problems.png'
    plt.savefig(image_path, bbox_inches='tight')

    # Show the plot
    plt.show()
