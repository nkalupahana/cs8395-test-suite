import os
import json
import matplotlib.pyplot as plt

# Path to the directory containing the score files
scores_directory = 'scores'
all_similarities = []

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
            # Check if 'similarities' exists and has at least one valid entry
            similarities = data.get('similarities', [])
            valid_similarities = [sim for sim in similarities if sim is not None]
            if valid_similarities:
                all_similarities.append(valid_similarities)
                problem_labels.append(filename.replace('.json', ''))  # Remove '.json' from the filename for the label

# Check if we have data to plot
if not all_similarities:
    print("No data to plot.")
else:
    # Create a boxplot of the similarities for all problems
    plt.figure(figsize=(10, 6))
    plt.boxplot(all_similarities, vert=True, patch_artist=True)  # Vert sets the boxplot orientation to vertical
    plt.title('Jaccard Similarity Boxplot for All Problems')
    plt.ylabel('Jaccard Similarity Score')

    # Add custom x-axis labels based on the problem names
    plt.xticks(range(1, len(problem_labels)+1), problem_labels, rotation=45, ha='right')

    image_path = 'jaccard_boxplot_all_problems.png'
    plt.savefig(image_path, bbox_inches='tight')

    # Optionally remove the empty slots from the x-axis
    plt.xlim(0.5, len(problem_labels) + 0.5)

    # Show the plot
    plt.show()
