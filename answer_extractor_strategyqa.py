import pandas as pd


def process_xlsx(input_file, output_file):

    # Load the input xlsx file
    df = pd.read_excel(input_file, header=None)

    # Create a new DataFrame to store results
    result_df = pd.DataFrame(columns=[0])

    # Iterate through each row in the first column
    for index, cell_value in df[0].items():
        if pd.isna(cell_value):  # Check if cell is empty
            result = ""
        else:
            # Convert cell value to string (if not already) and find first occurrence of "True" or "False"
            text = str(cell_value)
            true_index = text.find("True")
            false_index = text.find("False")

            # Determine which appears first
            if true_index == -1 and false_index == -1:
                result = ""
            elif true_index != -1 and (false_index == -1 or true_index < false_index):
                result = "True"
            else:
                result = "False"

        # Add the result to the result DataFrame
        result_df.loc[index] = [result]

    # Save the result DataFrame to a new xlsx file
    result_df.to_excel(output_file, index=False, header=False)


if __name__ == "__main__":

    # Modify the file paths as needed
    input_file = 'strategyqa_llama_responses.xlsx'
    output_file = 'strategyqa_llama_extracted_answers.xlsx'
    process_xlsx(input_file, output_file)