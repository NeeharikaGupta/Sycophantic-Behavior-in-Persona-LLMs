import pandas as pd
import re


# Use regex to find the first numeric value (integer or float)
def extract_first_numeric_value(input_string):

    # Check if the input string is empty or NaN
    if not input_string or input_string.lower() == 'nan':
        return 0

    # Use regex to find the first numeric value with optional commas and decimals
    match = re.search(r'\d{1,3}(?:,\d{3})*(\.\d+)?', input_string)
    if match:
        # Remove commas before returning the matched numeric value
        return match.group(0).replace(',', '')
    return 0


# Use regex to find the last numeric value (integer or float)
def extract_last_numeric_value(input_string):

    # Check if the input string is empty or NaN
    if not input_string or pd.isna(input_string):
        return 0

    # Use regex to find all numeric values, ignoring commas within numbers
    matches = re.findall(r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b', input_string)

    # Check if any matches were found
    if matches:
        return matches[-1].replace(',', '')  # Return the last match without commas
    return 0


def process_xlsx(input_file, output_file):

    # Read the Excel file
    df = pd.read_excel(input_file, header=None)

    # We extract the first and last number from the response and consider both for answer matching
    df['First Extracted Number'] = df.iloc[0:, 0].apply(lambda x: extract_first_numeric_value(str(x)))
    df['Last Extracted Number'] = df.iloc[0:, 0].apply(lambda x: extract_last_numeric_value(str(x)))
    output_df = df[['First Extracted Number', 'Last Extracted Number']]

    # Save the result DataFrame to a new xlsx file
    output_df.to_excel(output_file, index=False)


if __name__ == "__main__":

    # Modify the file paths as needed
    input_file = 'gsm8k_llama_responses.xlsx'
    output_file = 'gsm8k_llama_extracted_answers.xlsx'
    process_xlsx(input_file, output_file)