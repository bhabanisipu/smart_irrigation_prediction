import pandas as pd

def load_data(file_path):
    """
    Load the raw irrigation dataset
    """
    try:
        df = pd.read_excel(file_path)
        print("Dataset loaded sucessfully.")
        print("shape of the dataset is : ",df.shape)

        return df
    except FileNotFoundError as e:
        print("The file is not found : ",e)
        raise
    except pd.errors.EmptyDataError as e:
        print("The Dataset is empty : ",e)
        raise
    except Exception as e:
        print("The error will occur during data load time.")
        raise