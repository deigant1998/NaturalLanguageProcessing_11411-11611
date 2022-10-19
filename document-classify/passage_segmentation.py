import stanza

stanza.download('en')
nlp = stanza.Pipeline(lang='en', processors='tokenize')

output_file = open("/Users/deigant/Desktop/CMU/CMU Course Materials/11-611/hw1/HW01/documents/indian_states/Uttar_Pradesh_new.txt", "w");
output_file_1 = open("/Users/deigant/Desktop/CMU/CMU Course Materials/11-611/hw1/HW01/documents/indian_states/Uttar_Pradesh_new1.txt", "w");

universal_count = 0;

with open("/Users/deigant/Desktop/CMU/CMU Course Materials/11-611/hw1/HW01/documents/indian_states/Uttar_Pradesh.txt") as file:
    count = 0;
    for group_no, group_lines in enumerate(file):
        lines = nlp(group_lines)
        for line_ in lines.sentences:
            line = line_.text
            if len(line.strip()) == 0:
                continue;
            output_file_1.write(str(count) + ": " + line.strip() + "\n");
            output_file.write(line.strip() + "\n");
            universal_count += 1;
            count += 1;
            if(count % 10 == 0):
                output_file.write("\n \n \n \n");
                output_file_1.write("\n \n \n \n");
                count = 0;
        