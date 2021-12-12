# helper functions to preprocess datasets
from constants import START_TAG, STOP_TAG

def _preprocess_training_file1(file_path):
    with open(file_path, encoding="utf8") as f:
        data = f.read().splitlines()
        data[:] = [x for x in data if x]

    output = []
    for i in data:
        i = i.split(" ")
        if(len(i) > 2):
            i = [" ".join(i[0:len(i)-1]), i[len(i)-1]]
            output.append(i)
        else:
            output.append(i)
        
    return output

def _preprocess_testing_file1(path):
    with open(path, encoding="utf8") as f:
        data = f.read().splitlines()

    output = []
    for word in data:
        output.append(word)

    return output

def _preprocess_training_file2(training_file):
    tags = []
    tags_with_start_stop = []
    words = []

    with open(training_file,"r",encoding="utf8") as f:
        document = f.read().rstrip()
        lines = document.split("\n\n")

        for line in lines:
            tags_list = []
            tags_with_start_stop_list = []
            words_list = []

            for word_tag in line.split("\n"):
                i = word_tag.split(" ")

                if len(i) > 2:
                    i = [" ".join(i[0:len(i)-1]), i[len(i)-1]]

                word, tag = i[0], i[1]
                words_list.append(word)
                tags_list.append(tag)

            tags.append(tags_list)
            tags_with_start_stop_list = [START_TAG] + tags_list + [STOP_TAG]
            tags_with_start_stop.append(tags_with_start_stop_list)
            words.append(words_list)
    
    return tags, tags_with_start_stop, words

def _preprocess_test_file2(testing_file):
    test_words = []

    with open(testing_file, encoding="utf8") as f:
        document = f.read().rstrip()
        lines = document.split("\n\n")

        for line in lines:
            word_list = []
            for word in line.split("\n"):
                word_list.append(word)
            test_words.append(word_list)

    return test_words