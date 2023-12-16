import os
import json
import matplotlib.pyplot as plt

# Path to the directory containing the score files
scores_directory = 'scores'
all_similarities = {0.2: [], 0.5: [], 0.7: [], 1.0: []}  # Dictionary to store similarities for each temperature

# Problem labels for x-axis
problem_labels = []

# Iterate over each file in the scores directory
for filename in os.listdir(scores_directory):
    if filename.endswith('.json'):
        # Extract temperature from the filename if it follows the expected format
        parts = filename.split('_T')
        if len(parts) == 2:
            try:
                temperature = float(parts[1].split('.json')[0])
            except ValueError:
                continue  # Skip files with invalid temperature values
        else:
            continue  # Skip files with unexpected naming format
        
        # Construct the full file path
        file_path = os.path.join(scores_directory, filename)

        # Open and read the JSON file
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
            # Check if 'similarities' exists and has at least one valid entry
            similarities = data.get('similarities', [])
            valid_similarities = [sim for sim in similarities if sim is not None]
            if valid_similarities:
                all_similarities[temperature].extend(valid_similarities)
                problem_labels.append(filename.replace('.json', ''))  # Remove '.json' from the filename

# Check if we have data to plot
if not any(all_similarities.values()):
    print("No data to plot.")
else:
    # Create a boxplot of the similarities for each temperature
    plt.figure(figsize=(12, 6))
    boxplot_data = [all_similarities[temperature] for temperature in all_similarities.keys()]
    temperatures_to_plot = [str(temperature) for temperature in all_similarities.keys()]
    
    plt.boxplot(boxplot_data, vert=True, patch_artist=True, labels=temperatures_to_plot)
    plt.title('Jaccard Similarity Boxplot for Different Temperatures')
    plt.ylabel('Jaccard Similarity Score')
    plt.xlabel('Temperature')

    image_path = 'jaccard_boxplot_by_temperature.png'
    plt.savefig(image_path, bbox_inches='tight')

    # Optionally remove the empty slots from the x-axis
    plt.xticks(range(1, len(temperatures_to_plot) + 1), temperatures_to_plot, rotation=0, ha='center')

    # Show the plot
    plt.show()
