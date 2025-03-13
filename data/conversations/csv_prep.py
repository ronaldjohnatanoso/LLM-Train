import csv
import os

def csv_to_text(input_csv, output_txt):
    with open(input_csv, 'r', encoding='utf-8') as csv_file, open(output_txt, 'w', encoding='utf-8') as txt_file:
        reader = csv.reader(csv_file)
        # Skip header row if needed - uncomment the next line if your CSV has headers
        # next(reader)
        
        for row in reader:
            if len(row) >= 3:  # Make sure row has enough columns
                # Extract columns B and C (indices 1 and 2)
                sentence_b = row[1].strip()
                sentence_c = row[2].strip()
                
                # Write to text file as "sentence_b. sentence_c"
                txt_file.write(f"{sentence_b}. {sentence_c}\n")

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define input and output files in the same directory
input_file = os.path.join(current_dir, "conv.csv")
output_file = os.path.join(current_dir, "input.txt")

csv_to_text(input_file, output_file)