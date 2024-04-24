# read in 'name' and 'steps' columns from RAW_recipes.csv into a pd.DataFrame.

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
    print("preprocessing data...")
    data = preprocess("data/RAW_recipes.csv")

    print("saving data to data/processed_data.csv")
    data.to_csv("data/processed_data.csv", index=False)

    print("done!")
