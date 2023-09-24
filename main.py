from shutil import which, rmtree
import argparse
import glob
import itertools
import os
import subprocess
import json
from inspect import getsourcefile
from os.path import abspath

relpath = os.path.dirname(abspath(getsourcefile(lambda:0)))

# Check for vcstool
if which("vcs") is None:
    print("Run pip install vcstool to install vcs!")
    exit(1)

# Parse arguments
parser = argparse.ArgumentParser(description="Run LLM test suites")
parser.add_argument("--repos", help=".repos file to import", required=True)
parser.add_argument("--model", help="Model to run all test suites against", required=False)
args = parser.parse_args()

# Clone repos
rmtree("repos", ignore_errors=True)
os.mkdir("repos")
subprocess.Popen(f"vcs import --input {args.repos} repos", shell=True).wait()

# Run tests for each repo
outputs = []
for file in itertools.chain(glob.glob("repos/*/config.json")):
    directory = os.path.dirname(file)
    print(f"* {directory}")
    data = json.loads(open(file).read())
    if args.model is not None:
        data["model"] = args.model

    # Run scripts
    if "run_test" in data:
        print("Running test script!")
        subprocess.Popen(f"{data['run_test']} --model {data['model']} --relpath {relpath}", cwd=os.path.dirname(file), shell=True, stdout=subprocess.DEVNULL).wait()
    
    if "run_score" in data:
        print("Running scoring script!")
        subprocess.Popen(f"{data['run_score']} --model {data['model']} --relpath {relpath}", cwd=os.path.dirname(file), shell=True, stdout=subprocess.DEVNULL).wait()

    # Get test output
    output = json.loads(open(os.path.join(directory, "output.json")).read())
    output["tags"] = data["tags"]
    output["name"] = data["name"]
    outputs.append(output)

# Coalesce output
print(outputs)