import threading
import subprocess
import glob



# moving all data to one dir cd ~/Downloads/aclImdb/test/neg; tar -cf - * | (cd ~/Downloads/aclImdb/data; tar -xf -)


def process_files(files, thread_id):
    counter = 0

    for file in files :
        cleaned_text = ""
        cleaned_file = file.replace('/Users/henry/Downloads/aclImdb/data/', '/Volumes/USB/cs230/data_processed/')

        with open(file) as f_pointer:
            line = " ".join( f_pointer.readlines() )
            cleaned_text = line.replace("<br />", " ")

        if (len(cleaned_text) >= 50):
            with open(cleaned_file, 'w+') as fw:
                    fw.write(cleaned_text[:50]) # first 100 char

            #  1 US: Ava, 2 Australia: Karen, 3 India: Veena, 4 Scottish: Fiona, 5 South Africa: Tessa, 6 British: Serena

            output_file_1 = file.replace('/Users/henry/Downloads/aclImdb/data/', '/Volumes/USB/cs230/wav_1/').replace(".txt", ".wav")
            output_file_2 = file.replace('/Users/henry/Downloads/aclImdb/data/', '/Volumes/USB/cs230/wav_2/').replace(".txt", ".wav")
            output_file_3 = file.replace('/Users/henry/Downloads/aclImdb/data/', '/Volumes/USB/cs230/wav_3/').replace(".txt", ".wav")
            output_file_4 = file.replace('/Users/henry/Downloads/aclImdb/data/', '/Volumes/USB/cs230/wav_4/').replace(".txt", ".wav")
            output_file_5 = file.replace('/Users/henry/Downloads/aclImdb/data/', '/Volumes/USB/cs230/wav_5/').replace(".txt", ".wav")
            output_file_6 = file.replace('/Users/henry/Downloads/aclImdb/data/', '/Volumes/USB/cs230/wav_6/').replace(".txt", ".wav")


            subprocess.check_output(['say', '--file-format=WAVE', '--channels=1', '--data-format=LEF32@22050', '-f', cleaned_file, '-o', output_file_1, '-v', 'Ava', '-r', '175'])
            subprocess.check_output(['say', '--file-format=WAVE', '--channels=1', '--data-format=LEF32@22050', '-f', cleaned_file, '-o', output_file_2, '-v', 'Karen', '-r', '175'])
            subprocess.check_output(['say', '--file-format=WAVE', '--channels=1', '--data-format=LEF32@22050', '-f', cleaned_file, '-o', output_file_3, '-v', 'Veena', '-r', '175'])
            subprocess.check_output(['say', '--file-format=WAVE', '--channels=1', '--data-format=LEF32@22050', '-f', cleaned_file, '-o', output_file_4, '-v', 'Fiona', '-r', '175'])
            subprocess.check_output(['say', '--file-format=WAVE', '--channels=1', '--data-format=LEF32@22050', '-f', cleaned_file, '-o', output_file_5, '-v', 'Tessa', '-r', '175'])
            subprocess.check_output(['say', '--file-format=WAVE', '--channels=1', '--data-format=LEF32@22050', '-f', cleaned_file, '-o', output_file_6, '-v', 'Serena', '-r', '175'])

        counter += 1
        print(file + " done in " + str(thread_id) + " " + str(counter))


# 1:43pm on 480
if __name__ == "__main__":

    num_threads = 3
    thread_pointers = [i for i in range(num_threads)]

    files = glob.glob("/Users/henry/Downloads/aclImdb/data/*.txt")

    print("Processing " + str(len(files)) + " files:")

    files_per_part = int( len(files)  / num_threads )

    for i in range(num_threads):
        some_files = files[i * files_per_part : (i+1) * files_per_part]
        thread_pointers[i] = threading.Thread(target=process_files, args=(some_files, i))


    for i in range(num_threads):
        thread_pointers[i].start()

    for i in range(num_threads):
        thread_pointers[i].join()


    print("Done!")
