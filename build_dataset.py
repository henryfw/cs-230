
import threading
import subprocess
import glob
import re
import operator
import pickle
from audio import spec2wav, wav2spec, read_wav, write_wav
import matplotlib.pyplot as plt
import numpy as np

sr = 22050
n_fft = 512
win_length = 400
hop_length = 80
duration = 2  # sec

# moving all data to one dir cd ~/Downloads/aclImdb/test/neg; tar -cf - * | (cd ~/Downloads/aclImdb/data; tar -xf -)

def wav_to_spec_inverted(file):
    wav_x = read_wav(file, sr, duration)
    spec_x, _ = wav2spec(wav_x, n_fft, win_length, hop_length, False)

    spec_x_padding = np.array(spec_x[:, 0:300])
    spec_x_padding /= np.max(spec_x_padding)
    spec_x_padding.resize((257, 300))

    return np.swapaxes( spec_x_padding, 0, 1 )

def save_obj(obj, name ):
    with open('data/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('data/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def process_files(files, thread_id):
    counter = 0

    words = {}

    for file in files :

        with open(file) as f_pointer:
            line = " ".join( f_pointer.readlines() )
            cleaned_text = line.replace("<br />", " ").replace("/", " ").replace("\"", " ").replace("'", " ").replace("(", " ").replace("`", " ").replace("~", " ").replace("-", " ")
            cleaned_text = cleaned_text.replace(")", " ").replace(",", " ").replace("“", " ").replace("‘", " ").replace(".", " ").replace("?", " ").replace("!", " ")
            cleaned_text = cleaned_text.replace("$", " ")
            cleaned_text = cleaned_text.replace(":", " ").replace(";", " ").replace("_", " ").replace("--", " ").replace("{", " ").replace("}", " ").replace("=", " ")
            parts = cleaned_text.split(' ')
            for part in parts:
                part = part.strip().lower()
                use_this = True
                l = len(part)

                if l == 0 or l > 25: use_this = False
                elif part.startswith("http"): use_this = False
                elif "*" in part: use_this = False
                elif "--" in part: use_this = False
                elif "\x97" in part: use_this = False
                elif "\x85" in part: use_this = False

                if use_this:
                    if part in words:
                        words[part] += 1
                    else:
                        words[part] = 1


        counter += 1


    # sorted_words = sorted(words.items(), key=operator.itemgetter(1))
    #
    # print( sorted_words[-100:])
    # print( sorted_words[:200])


    data1 = []
    data2 = []


    counter = 0

    words_to_use = []

    save_part_index = 0

    for word in words:
        count = words[word]
        if count > 1:
            try:
                subprocess.check_output(
                    ['say', '--file-format=WAVE', '--channels=1', '--data-format=LEF32@22050', '-o', '/tmp/1.wav', '-v', 'Ava', '-r', '175', word])

                subprocess.check_output(
                    ['say', '--file-format=WAVE', '--channels=1', '--data-format=LEF32@22050', '-o', '/tmp/2.wav', '-v', 'Serena', '-r', '175', word])

                spec1 = wav_to_spec_inverted('/tmp/1.wav')
                spec2 = wav_to_spec_inverted('/tmp/2.wav')

                data1.append(spec1)
                data2.append(spec2)

                words_to_use.append(count)
                counter += 1
            except:
                pass


        if counter % 5000 == 0:
            print("counter " + str(counter))

            save_obj(words_to_use, "word_count_" + str(save_part_index))
            np.save("data/data1_" + str(save_part_index), np.array(data1))
            np.save("data/data2_" + str(save_part_index), np.array(data2))
            data1 = []
            data2 = []
            words_to_use = []
            save_part_index += 1

    if len(data1) > 0:
        save_obj(words_to_use, "word_count_" + str(save_part_index))
        np.save("data/data1_" + str(save_part_index), np.array(data1))
        np.save("data/data2_" + str(save_part_index), np.array(data2))

    print("total words " + str(counter))



if __name__ == "__main__":

    num_threads = 1
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
