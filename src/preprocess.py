import csv
import json


def preprocess_reclor(inputFile, outputFile):
    """
    The preprocess_reclor function takes in two arguments:
        1. inputFile - the name of the file containing ReCLoR data (e.g., reclor_train.json)
        2. outputFile - the name of a CSV file to write preprocessed data to (e.g., reclor_train_preprocessed)

    :param inputFile: Specify the input file name
    :param outputFile: Specify the file name of the output csv file
    :return: A csv file with the following columns:
    """
    myMap = {0: "A", 1: "B", 2: "C", 3: "D"}

    with open(inputFile, "r") as file:
        data = json.load(file)

    with open(outputFile, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["query", "response"])
        for item in data:
            allOptions = [f"{myMap[i]}. {val}" for i, val in enumerate(item["answers"])]
            query = "\n".join([item["context"], item["question"], *allOptions])
            response = allOptions[item["label"]]
            writer.writerow([query, response])


# Preprocess the LogiQA dataset
# Only includes the letter in the response
def filter_response(inputFile, outputFile):
    """
    The filter_response function takes in a file name and an output file name.
    It then reads the input file, strips all of the lines, and writes them to
    the output file as a CSV with two columns: query and response. The query is
    made up of 6 lines from the original text document (which are concatenated)
    and the response is one line above that.

    :param inputFile: Specify the file that you want to read from
    :param outputFile: Specify the name of the file that will be created
    :return: A csv file with the query and response
    """
    print(f"Saving to {outputFile}")
    with open(inputFile, "r", encoding="utf-8") as file:
        allLines = file.readlines()

    for i, line in enumerate(allLines):
        allLines[i] = line.strip()

    with open(outputFile, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["query", "response"])
        count = 8
        while count < len(allLines):
            query = "\n".join(allLines[count - 6 : count])
            response = allLines[count - 7].upper()
            writer.writerow([query, response])
            count += 8


# Preprocess the LogiQA dataset
# includes the full option choice in the response
def preprocess_logiqa(inputFile, outputFile):
    """
    The preprocess_logiqa function takes in a file path to the logiQA dataset and outputs a csv file with two columns:
        query - The question asked by the user.
        response - The correct answer to the question.

    :param inputFile: Specify the file to be read
    :param outputFile: Specify the name of the file to save to
    :return: A csv file with two columns: query and response
    """
    print(f"Saving to {outputFile}")
    with open(inputFile, "r", encoding="utf-8") as file:
        allLines = file.readlines()

    for i, line in enumerate(allLines):
        allLines[i] = line.strip()

    with open(outputFile, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["query", "response"])
        count = 8
        while count < len(allLines):
            choices = allLines[count - 4 : count]
            answer = allLines[count - 7].upper()
            index = ord(answer) - ord("A")
            writer.writerow(["\n".join(allLines[count - 6 : count]), choices[index]])
            count += 8


def preprocess_logiqa2(inputFile, outputFile):
    """
    The preprocess_logiqa2 function takes in a JSON file and outputs a CSV file.
    The input JSON file is the LogiQA dataset, which contains questions, answers, and options for each question.
    The output CSV file has two columns: query (which contains the question text) and response (which contains the answer).


    :param inputFile: Specify the input file path
    :param outputFile: Specify the name of the file that will be created
    :return: A csv file with the following format:
    """
    myMap = {0: "A", 1: "B", 2: "C", 3: "D"}
    with open(inputFile, "r") as file:
        lines = file.readlines()

    data = [json.loads(line) for line in lines]

    with open(outputFile, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["query", "response"])
        for item in data:
            allOptions = [f"{myMap[i]}. {val}" for i, val in enumerate(item["options"])]
            query = "\n".join([item["text"], item["question"], *allOptions])
            response = allOptions[item["answer"]]
            writer.writerow([query, response])
