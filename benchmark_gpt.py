# This code is used to call openAI's API to benchmark datasets

import os
from openai import OpenAI
import pandas as pd
from tqdm.auto import tqdm

# Initialize the client
client = OpenAI()


def gpt(question: str) -> str:
    """
    The gpt function takes a question as an argument and returns the answer to that question.
        The function uses OpenAI's GPT-3.5 Turbo API to generate answers for multiple choice questions.

    :param question: Pass in the question that you want to answer
    :return: A string
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You answer multiple choice questions with the correct letter answer. Your answer should be in this format: 'Correct Answer: {Letter}. Explanation: {Explanation}'",
            },
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content


def checkResponse(response: str, answer: str) -> int:
    """
    The checkResponse function takes two arguments:
        response - a string containing the generated answer to a question
        answer   - the correct answer to that question (a single character)

    :param response: The generated response
    :param answer: The correct answer from the dataset
    :return: An integer. -1 if invalid, 1 if correct, 0 if incorrect
    """
    response = response.strip()
    if response.lower()[:15] == "correct answer:":
        if response[16] == answer:
            return 1
        else:
            return 0
    return -1


def main() -> None:
    # set the openAI key
    os.environ["OPENAI_API_KEY"] = "<your-api-key-goes-here>"

    # Set up tqdm
    tqdm.pandas(desc="Processing queries")

    # Read in the csv file
    df = pd.read_csv("logicQA-test.csv")

    # Answer questions using the API, storing answers in a new column called gpt
    df["gpt"] = df["query"].progress_apply(gpt)

    # Grade the answers, storing grades 0, -1, and 1 in a new column called correct
    df["correct"] = df.apply(
        lambda row: checkResponse(row["gpt"], row["response"][0]), axis=1
    )

    # Count the results
    correct = (df["correct"] == 1).sum()
    invalid = (df["correct"] == -1).sum()
    total = df.shape[0]

    # Display the results
    print(f"Accuracy: {correct}/{total} ({correct / total * 100:.2f}%)")
    print(f"Invalid: {invalid}/{total} ({invalid / total * 100:.2f}%)")

    # Save the dataframe
    df.to_csv("gpt-3.5-outputs.csv")


if __name__ == "__main__":
    main()
