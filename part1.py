from constants import (
    ES_TRAIN_DATA_FILE,
    ES_TEST_DATA_FILE,
    ES_ACTUAL_DATA_FILE,
    ES_PART1_PREDICTED_DATA_FILE,
    RU_TRAIN_DATA_FILE,
    RU_TEST_DATA_FILE,
    RU_ACTUAL_DATA_FILE,
    RU_PART1_PREDICTED_DATA_FILE,
    UNK_WORD,
)
from evaluate import evaluateScores
from preprocessing import _preprocess_training_file1, _preprocess_testing_file1


def get_emission_using_MLE(training, k=1):
    tags = {}
    tags_to_word = {}
    emission = {}
    for data in training:
        word, tag = data[0], data[1]
        if tag in tags:
            tags[tag] += 1
        else:
            tags[tag] = 1

        tag_to_word = (tag, word)
        if tag_to_word in tags_to_word:
            tags_to_word[tag_to_word] += 1
        else:
            tags_to_word[tag_to_word] = 1

    for key in tags_to_word.keys():
        emission[key] = tags_to_word[key] / (tags[key[0]] + k)
    for key in tags.keys():
        transition = (key, UNK_WORD)
        emission[transition] = k / (tags[key] + k)

    return emission


def get_most_probable_tag(emission):
    highest_prob = {}
    output = {}
    for key, prob in emission.items():
        tag, word = key[0], key[1]
        if word not in highest_prob:
            highest_prob[word] = prob
            output[word] = tag
        else:
            if prob > highest_prob[word]:
                highest_prob[word] = prob
                output[word] = tag

    return output


def write_to_predicted_file_part(predicted_file, test_file, most_probable_tag):
    with open(predicted_file, "w", encoding="utf8") as f:
        for word in test_file:
            if len(word) > 0:
                try:
                    y = most_probable_tag[word]
                except:
                    y = most_probable_tag[UNK_WORD]
                f.write(f"{word} {y}\n")
            else:
                f.write("\n")


if __name__ == "__main__":
    # Part 1 code for ES Dataset
    ES_train_data = _preprocess_training_file1(ES_TRAIN_DATA_FILE)
    ES_test_data = _preprocess_testing_file1(ES_TEST_DATA_FILE)

    ES_emission_parameters = get_emission_using_MLE(ES_train_data)
    ES_most_probable_tag = get_most_probable_tag(ES_emission_parameters)
    write_to_predicted_file_part(
        ES_PART1_PREDICTED_DATA_FILE, ES_test_data, ES_most_probable_tag
    )

    # Part 1 code for RU Dataset
    RU_train_data = _preprocess_training_file1(RU_TRAIN_DATA_FILE)
    RU_test_data = _preprocess_testing_file1(RU_TEST_DATA_FILE)

    RU_emission_parameters = get_emission_using_MLE(RU_train_data)
    RU_most_probable_tag = get_most_probable_tag(RU_emission_parameters)
    write_to_predicted_file_part(
        RU_PART1_PREDICTED_DATA_FILE, RU_test_data, RU_most_probable_tag
    )

    print("\nResults for ES dataset")
    evaluateScores(ES_ACTUAL_DATA_FILE, ES_PART1_PREDICTED_DATA_FILE)

    print("\nResults for RU dataset")
    evaluateScores(RU_ACTUAL_DATA_FILE, RU_PART1_PREDICTED_DATA_FILE)
