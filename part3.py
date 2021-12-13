from constants import (
    ES_TRAIN_DATA_FILE,
    ES_TEST_DATA_FILE,
    ES_ACTUAL_DATA_FILE,
    ES_PART3_PREDICTED_DATA_FILE,
    RU_TRAIN_DATA_FILE,
    RU_TEST_DATA_FILE,
    RU_ACTUAL_DATA_FILE,
    RU_PART3_PREDICTED_DATA_FILE,
    START_TAG,
    STOP_TAG,
    UNK_WORD,
)
from evaluate import evaluateScores, write_to_predicted_file
from preprocessing import (
    _preprocess_training_file2,
    _preprocess_test_file2,
    _preprocess_training_file1,
)
from part1 import get_emission_using_MLE
from part2 import get_unique_elements, get_transition_pairs, get_transition_using_MLE

import collections
import math
import sys

word_output_list = []  # list of tuple(word, predicted_tag) for writing to file
viterbi_values = {}  # key: (n, tag)  value: float


def generate_viterbi_values(
    n,
    current_tag,
    word_list,
    words_unique,
    tags_unique,
    emission_params,
    transmission_params,
):
    global viterbi_values
    current_max_viterbi_value = -sys.float_info.max  # Smallest negative float

    if n == 0:
        return

    if n == 1:
        try:
            if word_list[n - 1] in words_unique:
                try:
                    current_max_viterbi_value = math.log(
                        emission_params[(current_tag, word_list[n - 1])]
                        * transmission_params[START_TAG][current_tag]
                    )
                except KeyError:
                    current_max_viterbi_value = -sys.float_info.max
            else:
                current_max_viterbi_value = math.log(
                    emission_params[(current_tag, UNK_WORD)]
                    * transmission_params[START_TAG][current_tag]
                )
        except ValueError:
            current_max_viterbi_value = -sys.float_info.max

        viterbi_values[(n, current_tag)] = current_max_viterbi_value
        return

    # Recursive call to generate viterbi_values for (n-1, tag)
    for tag in tags_unique:
        if (n - 1, tag) not in viterbi_values:
            generate_viterbi_values(
                n - 1,
                tag,
                word_list,
                words_unique,
                tags_unique,
                emission_params,
                transmission_params,
            )

    # Use viterbi values from n-1 to generate current viterbi value
    for tag in tags_unique:
        # Here, we use a try-except block because our emission parameters only contain emissions which appeared in our datasets
        # Thus, any unobserved emission will throw a KeyError, however it's value should be -inf, so we just catch the Error and proceed to the next tag
        # If transmission_params gives 0, math.log will throw a valueError, thus we catch it and skip the current tag since 0 means we should never consider it
        try:
            if word_list[n - 1] in words_unique:
                try:
                    value = viterbi_values[(n - 1, tag)] + math.log(
                        emission_params[(current_tag, word_list[n - 1])]
                        * transmission_params[tag][current_tag]
                    )
                except KeyError:
                    continue
            else:
                value = viterbi_values[(n - 1, tag)] + math.log(
                    emission_params[(current_tag, UNK_WORD)]
                    * transmission_params[tag][current_tag]
                )
        except ValueError:
            continue

        current_max_viterbi_value = max(current_max_viterbi_value, value)

    viterbi_values[(n, current_tag)] = current_max_viterbi_value


# function to kickstart viterbi recursive chain, and add (n+1, STOP) to veterbi_values
def start_viterbi(
    word_list, words_unique, tags_unique, emission_params, transmission_params
):
    global viterbi_values
    max_final_viterbi_value = -sys.float_info.max

    n = len(word_list)

    # Recursive call to generate viterbi_values for (n, tag)
    for tag in tags_unique:
        generate_viterbi_values(
            n,
            tag,
            word_list,
            words_unique,
            tags_unique,
            emission_params,
            transmission_params,
        )

    # Use viterbi values from n to generate viterbi value for (n+1, STOP)
    for tag in tags_unique:
        try:
            value = viterbi_values[(n, tag)] + math.log(
                transmission_params[tag][STOP_TAG]
            )
        except ValueError:
            continue
        max_final_viterbi_value = max(max_final_viterbi_value, value)

    viterbi_values[(n + 1, STOP_TAG)] = max_final_viterbi_value


def get_top_scores_from_dictionary(d, k=5):
    return collections.OrderedDict(sorted(d.items(), reverse=True)[:k])


def generate_predictions_viterbi(word_list, tags_unique, transmission_params):
    global viterbi_values
    total_viterbi_scores = {}
    n = len(word_list)

    for tag in tags_unique:
        try:
            value = viterbi_values[(n, tag)] + math.log(
                transmission_params[tag][STOP_TAG]
            )
            total_viterbi_scores[value] = [tag]
        except ValueError:
            continue

    total_viterbi_scores = get_top_scores_from_dictionary(total_viterbi_scores)

    # Generate predictions starting from n-1 going down to 1
    for i in range(n - 1, 0, -1):
        link = {}
        for tags in total_viterbi_scores.values():

            for tag in tags_unique:
                try:
                    value = viterbi_values[(i, tag)] + math.log(
                        transmission_params[tag][tags[0]]
                    )  # we take the first tag because we are working backwards
                    link[value] = [tag] + tags
                except ValueError:
                    continue

        total_viterbi_scores = get_top_scores_from_dictionary(link)
    return list(total_viterbi_scores.values())[-1]


if __name__ == "__main__":

    ES_train_data = _preprocess_training_file1(ES_TRAIN_DATA_FILE)
    ES_emission_parameters = get_emission_using_MLE(ES_train_data)

    ES_tags, ES_tags_with_start_stop, ES_train_words = _preprocess_training_file2(
        ES_TRAIN_DATA_FILE
    )
    ES_test_words = _preprocess_test_file2(ES_TEST_DATA_FILE)
    ES_unique_words = get_unique_elements(ES_train_words)
    ES_unique_tags = get_unique_elements(ES_tags)

    ES_transition_pair_count = get_transition_pairs(ES_tags_with_start_stop)
    ES_transition_parameters = get_transition_using_MLE(
        ES_unique_tags, ES_transition_pair_count, ES_tags_with_start_stop
    )

    RU_train_data = _preprocess_training_file1(RU_TRAIN_DATA_FILE)
    RU_emission_parameters = get_emission_using_MLE(RU_train_data)

    RU_tags, RU_tags_with_start_stop, RU_train_words = _preprocess_training_file2(
        RU_TRAIN_DATA_FILE
    )
    RU_test_words = _preprocess_test_file2(RU_TEST_DATA_FILE)
    RU_unique_words = get_unique_elements(RU_train_words)
    RU_unique_tags = get_unique_elements(RU_tags)

    RU_transition_pair_count = get_transition_pairs(RU_tags_with_start_stop)
    RU_transition_parameters = get_transition_using_MLE(
        RU_unique_tags, RU_transition_pair_count, RU_tags_with_start_stop
    )

    # Run and output Viterbi for ES
    ES_predicted_tags_list = []
    for word in ES_test_words:
        viterbi_values = {}
        start_viterbi(
            word,
            ES_unique_words,
            ES_unique_tags,
            ES_emission_parameters,
            ES_transition_parameters,
        )
        ES_generated_tag_list = generate_predictions_viterbi(
            word, ES_unique_tags, ES_transition_parameters
        )
        ES_predicted_tags_list.append(ES_generated_tag_list)

    write_to_predicted_file(
        ES_PART3_PREDICTED_DATA_FILE, ES_test_words, ES_predicted_tags_list
    )

    # Run and output Viterbi for RU
    RU_predicted_tags_list = []
    for word in RU_test_words:
        viterbi_values = {}
        start_viterbi(
            word,
            RU_unique_words,
            RU_unique_tags,
            RU_emission_parameters,
            RU_transition_parameters,
        )
        RU_generated_tag_list = generate_predictions_viterbi(
            word, RU_unique_tags, RU_transition_parameters
        )
        RU_predicted_tags_list.append(RU_generated_tag_list)

    write_to_predicted_file(
        RU_PART3_PREDICTED_DATA_FILE, RU_test_words, RU_predicted_tags_list
    )

    print("\nResults for ES dataset")
    evaluateScores(ES_ACTUAL_DATA_FILE, ES_PART3_PREDICTED_DATA_FILE)

    print("\nResults for RU dataset")
    evaluateScores(RU_ACTUAL_DATA_FILE, RU_PART3_PREDICTED_DATA_FILE)
