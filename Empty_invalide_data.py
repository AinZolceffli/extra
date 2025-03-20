import pandas as pd
import re
import os
import numpy as np

def parse_variables_file(filepath):
    """
    Parse the variables.txt file to extract columns to delete.
    
    Args:
        filepath (str): Path to the variables.txt file

    Returns:
        list: List of columns to delete
    """
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()

    delete_columns = []

    # Split content by variable definitions (each variable starts at beginning of line followed by newline)
    pattern = r'(?:^|\n)(\w+)\s*\n'
    variable_sections = re.split(pattern, content)[1:]  # Skip the first empty split

    # Process variable sections in pairs (name, content)
    for i in range(0, len(variable_sections), 2):
        if i + 1 >= len(variable_sections):
            break

        var_name = variable_sections[i].strip().lower()
        section = variable_sections[i + 1].strip()

        # Skip empty variable names
        if not var_name:
            continue

        # Check for DELETE instruction
        if "DELETE THIS COLUMN" in section:
            delete_columns.append(var_name)

    print(f"Columns to delete: {delete_columns}")

    return delete_columns


def clean_dataset(df, delete_columns):
    """
    Clean a dataset by replacing all invalid values with NaN and deleting specified columns.

    Args:
        df (pandas.DataFrame): The dataset to clean
        delete_columns (list): List of columns to delete

    Returns:
        pandas.DataFrame: Cleaned dataset
    """
    cleaned_df = df.copy()

    # Delete specified columns
    for column in [col for col in cleaned_df.columns if col.lower() in delete_columns]:
        print(f"Dropping column: {column}")
        cleaned_df.drop(column, axis=1, inplace=True)

    # Convert all non-numeric invalid values to NaN
    invalid_values = {"refusal", "don't know", "no answer", "other", "not applicable", "invalid", "blank", "not available", "66", "88"}
    
    for column in cleaned_df.columns:
        cleaned_df[column] = cleaned_df[column].apply(lambda x: np.nan if str(x).strip().lower() in invalid_values else x)

    print(f"\nFinal dataset has {cleaned_df.shape[0]} rows and {cleaned_df.shape[1]} columns.")

    return cleaned_df


def main(variables_file, data_file, output_file=None):
    """
    Main function to clean a dataset by deleting specified columns and replacing invalid values.

    Args:
        variables_file (str): Path to the variables.txt file
        data_file (str): Path to the dataset file
        output_file (str, optional): Path to save the cleaned dataset
    """
    print(f"Parsing variables file: {variables_file}")
    delete_columns = parse_variables_file(variables_file)

    # Determine file format and read the dataset
    file_ext = os.path.splitext(data_file)[1].lower()

    if file_ext == '.csv':
        df = pd.read_csv(data_file, dtype=str)  # Read as string to avoid auto-conversion issues
    elif file_ext in ['.xlsx', '.xls']:
        df = pd.read_excel(data_file)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")

    print(f"\nLoaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")

    # Clean the dataset
    print("\nCleaning dataset...")
    cleaned_df = clean_dataset(df, delete_columns)

    # Save the cleaned dataset if output file is provided
    if output_file:
        file_ext = os.path.splitext(output_file)[1].lower()

        if file_ext == '.csv':
            cleaned_df.to_csv(output_file, index=False)
        elif file_ext in ['.xlsx', '.xls']:
            cleaned_df.to_excel(output_file, index=False)
        else:
            raise ValueError(f"Unsupported output file format: {output_file}")

        print(f"\nCleaned dataset saved to: {output_file}")

    return cleaned_df


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        import argparse

        parser = argparse.ArgumentParser(description='Clean dataset by deleting columns and invalid values')
        parser.add_argument('variables_file', help='Path to the variables.txt file')
        parser.add_argument('data_file', help='Path to the dataset file (CSV or Excel)')
        parser.add_argument('--output', help='Path to save the cleaned dataset')

        args = parser.parse_args()

        main(args.variables_file, args.data_file, args.output)
    else:
        cleaned_df = main('ESS11-codebook-cleaned.txt', 'ESS11.csv', 'cleaned_dataES11_allVariables.csv')
