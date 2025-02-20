import csv
import os

# Get the directory of the currently running script
script_directory = os.path.dirname(os.path.abspath(__file__))

# Define the input and output file paths
input_csv = os.path.join(script_directory, 'conv.csv')
output_txt = os.path.join(script_directory, 'input.txt')

# Open the input CSV file and output text file
with open(input_csv, 'r', newline='', encoding='utf-8') as csvfile, open(output_txt, 'w', encoding='utf-8') as txtfile:
    csvreader = csv.reader(csvfile)
    
    # Skip the header row if there is one
    next(csvreader, None)
    
    # Iterate through each row in the CSV file
    for row in csvreader:
        # Extract the non-empty columns
        non_empty_columns = [col for col in row if col.strip()]
        
        # Write each non-empty column to the text file
        for sentence in non_empty_columns:
            txtfile.write(sentence + '\n')