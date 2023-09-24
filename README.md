# Test Suite Runner
*Made for Fall 2023 CS 8395-08*

This is a test suite runner for LLM assessments. 

## Installation

```
pip3 install -r requirements.txt
```

## Usage

```
usage: main.py [-h] --repos REPOS [--model MODEL] [--config-view] [--tags TAGS [TAGS ...]]

Run LLM test suites

options:
  -h, --help            show this help message and exit
  --repos REPOS         .repos file to import
  --model MODEL         Model to run all test suites against
  --config-view         Without running tests, get config breakdown of all repos
  --tags TAGS [TAGS ...]
                        Use repos that have one or more of these tags (space separated)
```

An example `.repos` file can be viewed at `sample.repos`.

## Test Suite Configuration

