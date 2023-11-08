import json
#import psutil
from timeit import default_timer as timer
from matrix_problems_cpu_for_loop_solutions import *
from optimized_matrix_functions_numpy import *
from optimized_matrix_functions_gpu_pytorch import *
from optimized_matrix_functions_gpu_cupy import *
#from matrix_problems_gpu_solutions import *
#from matrix_problems_torch_gpu_solutions import *

def load_problems(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def generate_solution(problem):
    # This is a placeholder. In a real-world scenario, you'd want to use a Generative AI model to generate the solution.
    solution = solve_problem_001_cpu
    # desc = problem["output_parameters"]
    if (problem["problem_id"]=="001"):
        # print("prime problem")
        solution = solve_problem_001_cpu, solve_problem_001_cpu_numpy, solve_problem_001_gpu_cupy, solve_problem_001_gpu_pytorch

        #solution = solve_problem_001
    if (problem["problem_id"]=="002"):
        # print("prime problem")
        solution = solve_problem_002_cpu, solve_problem_002_cpu_numpy, solve_problem_002_gpu_cupy, solve_problem_002_gpu_pytorch
        #solution = solve_problem_002
    if (problem["problem_id"]=="003"):
        # print("prime problem")
        solution = solve_problem_003_cpu, solve_problem_003_cpu_numpy, solve_problem_003_gpu_cupy, solve_problem_003_gpu_pytorch
        #solution = solve_problem_003
    if (problem["problem_id"] == "004"):
        # print("prime problem")
        solution = solve_problem_004_cpu, solve_problem_004_cpu_numpy, solve_problem_004_gpu_cupy, solve_problem_004_gpu_pytorch
        #solution = solve_problem_004
    if (problem["problem_id"] == "005"):
       # print("prime problem")
       solution = solve_problem_005_cpu, solve_problem_005_cpu_numpy, solve_problem_005_gpu_cupy, solve_problem_005_gpu_pytorch
       #solution = solve_problem_005
    if (problem["problem_id"] == "006"):
       # print("prime problem")
       solution = solve_problem_006_cpu, solve_problem_006_cpu_numpy, solve_problem_006_gpu_cupy, solve_problem_006_gpu_pytorch
       #solution = solve_problem_006
    if (problem["problem_id"] == "007"):
        # print("prime problem")
        solution = solve_problem_007_cpu, solve_problem_007_cpu_numpy, solve_problem_007_gpu_cupy, solve_problem_007_gpu_pytorch
        #solution = solve_problem_007
    if (problem["problem_id"] == "008"):
        # print("prime problem")
        solution = solve_problem_008_cpu, solve_problem_008_cpu_numpy, solve_problem_008_gpu_cupy, solve_problem_008_gpu_pytorch
        #solution = solve_problem_008
    if (problem["problem_id"] == "009"):
        # print("prime problem")
        solution = solve_problem_009_cpu, solve_problem_009_cpu_numpy, solve_problem_009_gpu_cupy, solve_problem_009_gpu_pytorch
        #solution = solve_problem_009
    if (problem["problem_id"] == "010"):
        # print("prime problem")
        solution = solve_problem_010_cpu, solve_problem_010_cpu_numpy, solve_problem_010_gpu_cupy, solve_problem_010_gpu_pytorch
        #solution = solve_problem_010

       # print("fibonacci")
    return solution


def evaluate_solution(problem, solution):
    # This is a basic evaluator. It assumes the AI's solution is a function that can be executed.
    # It will run the function with sample inputs and compare the result with expected outputs.
    passed_tests = 0
    total_tests = len(problem["sample_test_cases"])

    for test_case in problem["sample_test_cases"]:
        try:
            result = solution(*test_case["input"])
            # print("input")
            # print(test_case["input"])
            # print("result")
            # print(result)
            # print("expected_output")
            # print(test_case["expected_output"])
            # print(total_tests)
            if result == test_case["expected_output"]:
                passed_tests += 1

        except Exception as e:
            # If there's an error in the solution, it fails the test case
            pass

    return passed_tests, total_tests
def evaluate_solution_runtime(problem, solution):
    # This is a basic evaluator. It assumes the AI's solution is a function that can be executed.
    # It will run the function with sample inputs and compare the result with expected outputs.
    passed_tests = 0
    total_tests = len(problem["sample_test_cases"])
    print("\nproblem_id:", problem["problem_id"])
    start_time_0 = timer()
    for test_case in problem["sample_test_cases"]:
        #print("\nproblem_id:", problem["problem_id"])
        #print("\ninput:", test_case["input"])


        #print("\nexpected_output:", test_case["expected_output"])
        #print("\noutput:", solution[0](*test_case["input"]))
        try:

            result = solution[0](*test_case["input"])

            if [result] == test_case["expected_output"]:
                passed_tests += 1

        except Exception as e:
            # If there's an error in the solution, it fails the test case
            pass

    end_time_0 = timer()
    total_time_0 = end_time_0 -start_time_0
    #print("\ntotal_time:", total_time_0)
    #print("\nresult[0]:", result)

    start_time_1 = timer()
    for test_case in problem["sample_test_cases"]:
        #print("\nproblem_id:", problem["problem_id"])
        #print("\ninput:", test_case["input"])

        #print("\nexpected_output:", test_case["expected_output"])
        #print("\noutput:", solution[1](*test_case["input"]))
        try:

            result = solution[1](*test_case["input"])

            if [result] == test_case["expected_output"]:
                passed_tests += 1

        except Exception as e:
            # If there's an error in the solution, it fails the test case
            pass

    end_time_1 = timer()
    total_time_1 = end_time_1 - start_time_1
    #print("\ntotal_time:", total_time_1)


    start_time_2 = timer()
    for test_case in problem["sample_test_cases"]:
        #print("\nproblem_id:", problem["problem_id"])
        #print("\ninput:", test_case["input"])

        #print("\nexpected_output:", test_case["expected_output"])
        #print("\noutput:", solution[2](*test_case["input"]))
        try:

            result = solution[2](*test_case["input"])

            if [result] == test_case["expected_output"]:
                passed_tests += 1

        except Exception as e:
            # If there's an error in the solution, it fails the test case
            pass

    end_time_2 = timer()
    total_time_2 = end_time_2 - start_time_2
    #print("\ntotal_time:", total_time_2)

    start_time_3 = timer()
    for test_case in problem["sample_test_cases"]:
        #print("\nproblem_id:", problem["problem_id"])
        #print("\ninput:", test_case["input"])

        #print("\nexpected_output:", test_case["expected_output"])
        #print("\noutput:", solution[3](*test_case["input"]))
        try:

            result = solution[3](*test_case["input"])

            if [result] == test_case["expected_output"]:
                passed_tests += 1

        except Exception as e:
            # If there's an error in the solution, it fails the test case
            pass

    end_time_3 = timer()
    total_time_3 = end_time_3 - start_time_3
    print("\ntotal_time:", total_time_3)
    acc1 = total_time_0 / total_time_1
    acc2 = total_time_0 / total_time_2
    acc3 = total_time_0 / total_time_3

    print("acc1:", acc1)
    print("acc2:", acc2)
    print("acc3:", acc3)

    return total_time_0, passed_tests/2

def main():
    #problems = load_problems("updated_matrix_problems_after_replacement.json")
    problems = load_problems("all_updated_problems_100x100.json")

    results = []

    for problem in problems:
        solution = generate_solution(problem)
        runtime, peakmemory = evaluate_solution_runtime(problem, solution)
        results.append({
            "problem_id": problem["problem_id"],
            "runtime": runtime,
            "peakmemory": peakmemory
        })
    print(results)

    """
    for problem in problems:
        if(problem["problem_id"]) == "066": continue
        else:
            solution = generate_solution(problem)
            passed_tests, total_tests = evaluate_solution(problem, solution)
            results.append({
            "problem_id": problem["problem_id"],
            "passed_tests": passed_tests,
            "total_tests": total_tests
            }
            )


    # Print the results
    for result in results:
        print(f"Problem ID: {result['problem_id']}, Passed Tests: {result['passed_tests']}/{result['total_tests']}")
        print(f"Problem ID: {result['problem_id']}, Score: {passed_tests/total_tests}")
    """

"""
    # Example data to be written to the JSON file
    data = {
        "output": {
            "grade1": 0,
        }
    }

    # Writing JSON data
    with open('output.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print("JSON data has been written to 'output.json'")
"""

# """
if __name__ == "__main__":
    main()
