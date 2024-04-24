# read in 'name' and 'steps' columns from RAW_recipes.csv into a pd.DataFrame.

import sys

import pandas as pd


def read_data(file_path):
    data = pd.read_csv(file_path)
    data = data[["name", "steps"]]
    return data


def preprocess(file_path):
    data = read_data(file_path)

    # each element in the steps column is a list of strings. Please loop through the list and convert the list to one string, with each elemetn from the list numbers 1-(len(list)) followed by a period and a new line.
    data["steps"] = data["steps"].apply(
        lambda x: "\n".join(
            [str(i + 1) + ". " + step for i, step in enumerate(eval(x))]
        )
    )

    return data


if __name__ == "__main__":
    # use command line prompt to run this script given source and output file paths

    if len(sys.argv) != 3:
        print("Usage: python data_preprocess.py <source_file_path> <output_file_path>")
        sys.exit(1)

    source_file_path = sys.argv[1]
    output_file_path = sys.argv[2]

    print("preprocessing data...")
    data = preprocess(source_file_path)

    print(f"saving data to {output_file_path}")
    data.to_csv(output_file_path, index=False)

    print("done!")
