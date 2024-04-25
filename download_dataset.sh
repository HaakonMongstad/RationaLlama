# Make a temporary folder
mkdir temp
cd temp || exit

echo Downloading the dataset...

# Download the recipes dataset from kaggle
kaggle datasets download -d shuyangli94/food-com-recipes-and-user-interactions

# Unzip the dataset
unzip food-com-recipes-and-user-interactions.zip

# Move back to the parent directory
cd ..

echo Preprocessing the dataset...

# Preprocess the data
python preprocess.py ./temp/RAW_recipes.csv ./data/

# Remove the temporary folder
rm -r temp