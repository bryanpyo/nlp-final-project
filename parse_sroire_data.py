import json
import os
DELIMITER = '<$$$>'


def get_data_loc(data_folder, truth_folder, data_folder2, truth_folder2):
    with open('parsed_sroire_loc_tl_str.txt', 'w') as output:
        num_processed = 0
        for filename in os.listdir(data_folder):
            final_string = ""
            if num_processed%100 == 0:
                print(f'Have processed {num_processed} documents')
            data_file = os.path.join(data_folder, filename)
            if os.path.isfile(data_file):
                with open(data_file) as f:
                    for line in f:
                        text_list = line.split(',')[8:]
                        stringified = ','.join(text_list)
                        final_string += stringified[:-1] + ' ' # get rid of new line and add space
                        final_string += str(line.split(',')[0]) + ' ' + str(line.split(',')[1]) + ' '
            final_string = final_string[:-1]

            truth_file = os.path.join(truth_folder, filename)
            if os.path.isfile(truth_file):
                with open(truth_file) as f:
                    data = json.load(f)
                    for key, value in data.items():
                        datum = final_string + DELIMITER + key + DELIMITER + value + '\n'
                        output.write(datum)

            num_processed += 1

        #same as above but with the other dataset
        num_processed = 0
        for filename in os.listdir(data_folder2):
            final_string = ""
            if num_processed % 100 == 0:
                print(f'Have processed {num_processed} documents')
            data_file = os.path.join(data_folder2, filename)
            if os.path.isfile(data_file):
                with open(data_file) as f:
                    for line in f:
                        if len(line) > 1:
                            text_list = line.split(',')[8:]
                            stringified = ','.join(text_list)
                            final_string += stringified[:-1] + ' '
                            final_string += str(line.split(',')[0]) + ' ' + str(line.split(',')[1]) + ' '
            final_string = final_string[:-1]


            truth_file = os.path.join(truth_folder2, filename)
            if os.path.isfile(truth_file):
                with open(truth_file) as f:
                    data = json.load(f)
                    for key, value in data.items():
                        datum = final_string + DELIMITER + key + DELIMITER + value + '\n'
                        output.write(datum)

            num_processed += 1
def get_data_no_loc(data_folder, truth_folder, data_folder2, truth_folder2):
    with open('parsed_sroire.txt', 'w') as output:
        num_processed = 0
        for filename in os.listdir(data_folder):
            final_string = ""
            if num_processed%100 == 0:
                print(f'Have processed {num_processed} documents')
            data_file = os.path.join(data_folder, filename)
            if os.path.isfile(data_file):
                with open(data_file) as f:
                    for line in f:
                        text_list = line.split(',')[8:]
                        stringified = ','.join(text_list)
                        final_string += stringified[:-1] + ' '
            final_string = final_string[:-1]

            truth_file = os.path.join(truth_folder, filename)
            if os.path.isfile(truth_file):
                with open(truth_file) as f:
                    data = json.load(f)
                    for key, value in data.items():
                        datum = final_string + DELIMITER + key + DELIMITER + value + '\n'
                        output.write(datum)

            num_processed += 1

        #same sht as above but with the other dataset
        num_processed = 0
        for filename in os.listdir(data_folder2):
            final_string = ""
            if num_processed % 100 == 0:
                print(f'Have processed {num_processed} documents')
            data_file = os.path.join(data_folder2, filename)
            if os.path.isfile(data_file):
                with open(data_file) as f:
                    for line in f:
                        text_list = line.split(',')[8:]
                        stringified = ','.join(text_list)
                        final_string += stringified[:-1] + ' '
            final_string = final_string[:-1]

            truth_file = os.path.join(truth_folder2, filename)
            if os.path.isfile(truth_file):
                with open(truth_file) as f:
                    data = json.load(f)
                    for key, value in data.items():
                        datum = final_string + DELIMITER + key + DELIMITER + value + '\n'
                        output.write(datum)

            num_processed += 1


def main():
    get_data_loc(r"C:\Users\Bryan Pyo\Documents\SROIE2019\train\box", r"C:\Users\Bryan Pyo\Documents\SROIE2019\train\entities",
                    r"C:\Users\Bryan Pyo\Documents\SROIE2019\test\box", r"C:\Users\Bryan Pyo\Documents\SROIE2019\test\entities")



    # with open(r"C:\Users\Bryan Pyo\Documents\SROIE2019\train\entities\X00016469612.txt") as f:
    #     data = json.load(f)
    #     print(data)
    #
    # with open('parsed_sroire.txt', 'w') as f:
    #     f.write('Create a new text file!\n')
    #     f.write('Create a new text file!')


if __name__=='__main__':
    main()
