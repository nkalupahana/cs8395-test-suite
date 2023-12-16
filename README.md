# LLM Prompt Variation and Determinism Analysis Framework

This framework is designed to evaluate the consistency of a Large Language Model's (LLM) responses to various prompt variations. It includes scripts for generating prompt variations, invoking the LLM to get responses, and analyzing these responses to measure similarity and distance metrics. Generated with ChatGPT.

## Overview

The framework consists of two main Python scripts:

- `generate_prompts.py`: Loads base prompts, generates variations, collects responses from the LLM, and analyzes these responses to produce similarity and distance metrics.
- `test_scores.py`: Aggregates individual similarity and distance scores to calculate overall averages and outputs these to a JSON file.

## Prerequisites

Before running the framework, ensure the following prerequisites are met:

1. Python 3.6 or higher is installed.
2. Required Python packages are installed. You can install them using the following command:
    ```
    pip install openai numpy nltk sklearn python-Levenshtein
    ```
3. An OpenAI API key is obtained and set in the `generate_prompts.py` script.

## Setup

1. Clone or download this repository to your local machine.
2. Navigate to the framework's directory in your terminal.
3. Install the required Python packages using `pip` (see Prerequisites above).
4. Place your JSON files containing the base prompts in the `prompts` directory.

## Running the Framework

To run the framework, execute the following steps:

1. **Generate Prompt Variations and Analyze Responses:**
    ```
    python generate_prompts.py
    ```
    - This will process the base prompts, generate variations, fetch responses from the LLM, and save the results in the `responses` and `scores` directories.

2. **Calculate Average Scores:**
    ```
    python test_scores.py
    ```
    - After the above script has completed, this will aggregate the scores and compute average similarity and distance measures.

## Output

- The `responses` directory contains JSON files with the LLM's responses to both base and variant prompts.
- The `scores` directory holds the similarity and distance metrics for each prompt.
- The `output.json` file provides the aggregated average scores across all prompts.

## Customization

- You can customize the types of prompt variations and the evaluation metrics by modifying the `generate_variations`, `calculate_jaccard_similarity`, and `calculate_levenshtein_distance` functions in the `generate_prompts.py` script.

## Support

For any questions or issues, please open an issue on the GitHub repository page.
