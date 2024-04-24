import sys

import pandas as pd


def read_data(file_path):
    data = pd.read_csv(file_path)
    data = data[["name", "ingredients", "steps"]]
    return data


def preprocess(file_path):
    data = read_data(file_path)

    # create query column with name of recipe
    data["query"] = data["name"]

    # edit ingredients to be one string
    data["ingredients"] = data["ingredients"].apply(
        lambda x: "\n".join(
            [str(i + 1) + ". " + ingredient for i, ingredient in enumerate(eval(x))]
        )
    )

    # edit steps to be one string
    data["steps"] = data["steps"].apply(
        lambda x: "\n".join(
            [str(i + 1) + ". " + step for i, step in enumerate(eval(x))]
        )
    )

    # create response column with ingredients and steps
    data["response"] = data.apply(
        lambda row: f"You will need these ingredients: \n{row['ingredients']} \nHere are the steps to make the recipe: \n{row['steps']}\n",
        axis=1,
    )

    # drop name, ingredients, and steps columns
    data = data[["query", "response"]]

    return data


if __name__ == "__main__":
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
