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
            # Convert cell value to string (if not already) and find first occurrence of "A" or "B"
            text = str(cell_value)
            a_index = text.find("A")
            b_index = text.find("B")

            # Determine which appears first
            if a_index == -1 and b_index == -1:
                result = ""
            elif a_index != -1 and (b_index == -1 or a_index < b_index):
                result = "A"
            else:
                result = "B"

        # Add the result to the result DataFrame
        result_df.loc[index] = [result]

    # Save the result DataFrame to a new xlsx file
    result_df.to_excel(output_file, index=False, header=False)


if __name__ == "__main__":

    # Modify the file paths as needed
    input_file = 'quarel_llama_responses.xlsx'
    output_file = 'quarel_llama_extracted_answers.xlsx'
    process_xlsx(input_file, output_file)