from constants import (
    ES_TRAIN_DATA_FILE,
    ES_TEST_DATA_FILE,
    ES_ACTUAL_DATA_FILE,
    ES_PART4_PREDICTED_DATA_FILE,
    RU_TRAIN_DATA_FILE,
    RU_TEST_DATA_FILE,
    RU_ACTUAL_DATA_FILE,
    RU_PART4_PREDICTED_DATA_FILE,
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
from part2 import get_unique_elements, count_y

import math
import sys


def get_transition_pairs(tags):
    transition_pair_count = {}

    for tag in tags:
        for tag1, tag2 in zip(tag[:-1], tag[1:]):
            transition_pair = (tag1, tag2)
            if transition_pair in transition_pair_count:
                transition_pair_count[transition_pair] += 1
            else:
                transition_pair_count[transition_pair] = 1

    return transition_pair_count


def get_transition_triplets(tags):
    transition_triplet_count = {}

    for tag in tags:
        for tag1, tag2, tag3 in zip(tag[:-2], tag[1:-1], tag[2:]):
            transition_triplet = (tag1, tag2, tag3)
            if transition_triplet in transition_triplet_count:
                transition_triplet_count[transition_triplet] += 1
            else:
                transition_triplet_count[transition_triplet] = 1

    return transition_triplet_count


def get_transition_pairs_using_MLE(
    unique_tags, transition_pair_count, tags_with_start_stop
):

    unique_tags = [START_TAG] + unique_tags + [STOP_TAG]
    transition = {}
    for u in unique_tags[:-1]:  # omit STOP
        transition_row = {}
        for v in unique_tags[1:]:  # omit START
            transition_row[v] = 0.0
        transition[u] = transition_row

    # populate transition parameters with counts
    for u, v in transition_pair_count:
        transition[u][v] += transition_pair_count[(u, v)]

    # divide transition_count by count_yi, to get probability
    for u, transition_row in transition.items():
        count_yi = count_y(u, tags_with_start_stop)
        # words in training set
        for v, transition_count in transition_row.items():
            if count_yi == 0:
                transition[u][v] = 0.0
            else:
                transition[u][v] = transition_count / count_yi

    return transition


def get_transition_triplets_using_MLE(
    unique_tags, transition_triplet_count, tags_with_start_stop
):
    transition = {}
    unique_tags = [START_TAG] + unique_tags + [STOP_TAG]

    # Tags (u -> v -> w)
    for u in unique_tags[:-1]:  # omit STOP for first tag
        transition_first_tag = {}
        for v in unique_tags[1:-1]:  # omit START and STOP for second tag
            transition_second_tag = {}
            for w in unique_tags[1:]:  # omit START for third tag
                transition_second_tag[w] = 0.0
            transition_first_tag[v] = transition_second_tag
        transition[u] = transition_first_tag

    # fill up transition parameters
    for u, v, w in transition_triplet_count:
        transition[u][v][w] += transition_triplet_count[(u, v, w)]

    # divide transition_count by count_yi, to get probability
    for u, transition_first_tag in transition.items():
        for v, transition_second_tag in transition_first_tag.items():
            count_yi = count_y(u, tags_with_start_stop)

            for w, instance_in_training_set in transition_second_tag.items():
                if count_yi == 0:
                    transition[u][v][w] = 0.0
                else:
                    transition[u][v][w] = instance_in_training_set / count_yi

    return transition


word_output_list = []  # list of tuple(word, predicted_tag) for writing to file
viterbi_values = {}  # key: (n, tag)  value: float


def generate_viterbi_values(
    n,
    current_tag,
    word_list,
    words_unique,
    tags_unique,
    emission_params,
    transmission_pair_params,
    transmission_triplet_params,
):
    global viterbi_values
    current_max_viterbi_value = -sys.float_info.max  # Smallest negative float

    if n == 0:
        return

    # Use emission pair for n == 1
    if n == 1:
        try:
            if word_list[n - 1] in words_unique:
                try:
                    current_max_viterbi_value = math.log(
                        emission_params[(current_tag, word_list[n - 1])]
                        * transmission_pair_params[START_TAG][current_tag]
                    )
                except KeyError:
                    current_max_viterbi_value = -sys.float_info.max
            else:
                current_max_viterbi_value = math.log(
                    emission_params[(current_tag, UNK_WORD)]
                    * transmission_pair_params[START_TAG][current_tag]
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
                transmission_pair_params,
                transmission_triplet_params,
            )

    # Handle n == 2 where first tag in triplet is START
    if n == 2:
        for tag in tags_unique:
            try:
                if word_list[n - 1] in words_unique:
                    try:
                        value = viterbi_values[(n - 1, tag)] + math.log(
                            emission_params[(current_tag, word_list[n - 1])]
                            * transmission_triplet_params[START_TAG][tag][current_tag]
                        )
                    except KeyError:
                        continue
                else:
                    value = viterbi_values[(n - 1, tag)] + math.log(
                        emission_params[(current_tag, UNK_WORD)]
                        * transmission_triplet_params[START_TAG][tag][current_tag]
                    )
            except ValueError:
                continue
            current_max_viterbi_value = max(current_max_viterbi_value, value)

        viterbi_values[(n, current_tag)] = current_max_viterbi_value
        return

    # Use viterbi values from n-1 to generate current viterbi value
    for tag1 in tags_unique:
        for tag2 in tags_unique:
            try:
                if word_list[n - 1] in words_unique:
                    try:
                        value = viterbi_values[(n - 1, tag2)] + math.log(
                            emission_params[(current_tag, word_list[n - 1])]
                            * transmission_triplet_params[tag1][tag2][current_tag]
                        )
                    except KeyError:
                        continue
                else:
                    value = viterbi_values[(n - 1, tag2)] + math.log(
                        emission_params[(current_tag, UNK_WORD)]
                        * transmission_triplet_params[tag1][tag2][current_tag]
                    )
            except ValueError:
                continue

            current_max_viterbi_value = max(current_max_viterbi_value, value)

    viterbi_values[(n, current_tag)] = current_max_viterbi_value


# function to kickstart viterbi recursive chain, and add (n+1, STOP) to veterbi_values
def start_viterbi(
    word_list,
    words_unique,
    tags_unique,
    emission_params,
    transmission_pair_params,
    transmission_triplet_params,
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
            transmission_pair_params,
            transmission_triplet_params,
        )

    # Use viterbi values from n to generate viterbi value for (n+1, STOP)
    for tag1 in tags_unique:
        for tag2 in tags_unique:
            try:
                value = viterbi_values[(n, tag2)] + math.log(
                    transmission_triplet_params[tag1][tag2][STOP_TAG]
                )
            except ValueError:
                continue
            max_final_viterbi_value = max(max_final_viterbi_value, value)

    viterbi_values[(n + 1, STOP_TAG)] = max_final_viterbi_value


def generate_predictions_viterbi(word_list, tags_unique, transmission_triplet_params):
    global viterbi_values

    n = len(word_list)

    generated_tag_list = ["" for i in range(n)]

    current_best_tag = "O"
    current_best_tag_value = -sys.float_info.max

    # Handle case where word is length 1
    if n == 1:
        for tag in tags_unique:
            try:
                value = viterbi_values[(1, tag)] + math.log(
                    transmission_triplet_params[START_TAG][tag][STOP_TAG]
                )
            except ValueError:
                continue
            if value > current_best_tag_value:
                current_best_tag = tag
                current_best_tag_value = value

        generated_tag_list[0] = current_best_tag
        return generated_tag_list

    # Compute tag for n
    for tag1 in tags_unique:
        for tag2 in tags_unique:
            try:
                value = viterbi_values[(n, tag2)] + math.log(
                    transmission_triplet_params[tag1][tag2][STOP_TAG]
                )
            except ValueError:
                continue
            if value > current_best_tag_value:
                current_best_tag = tag2
                current_best_tag_value = value

    generated_tag_list[n - 1] = current_best_tag

    # Generate predictions starting from n-1 going down to 1
    for i in range(n - 1, 0, -1):
        current_best_tag = "O"
        current_best_tag_value = -sys.float_info.max

        for tag1 in tags_unique:
            for tag2 in tags_unique:
                try:
                    value = viterbi_values[(i, tag2)] + math.log(
                        transmission_triplet_params[tag1][tag2][generated_tag_list[i]]
                    )
                except ValueError:
                    continue
                if value > current_best_tag_value:
                    current_best_tag = tag2
                    current_best_tag_value = value

        generated_tag_list[i - 1] = current_best_tag

    return generated_tag_list


if __name__ == "__main__":

    ES_train_data = _preprocess_training_file1(ES_TRAIN_DATA_FILE)
    ES_emission_parameters = get_emission_using_MLE(ES_train_data)

    # Get ES tags
    ES_tags, ES_tags_with_start_stop, ES_train_words = _preprocess_training_file2(
        ES_TRAIN_DATA_FILE
    )
    ES_unique_tags = get_unique_elements(ES_tags)

    # Get ES words
    ES_test_words = _preprocess_test_file2(ES_TEST_DATA_FILE)
    ES_unique_words = get_unique_elements(ES_train_words)

    # Get ES transition pairs and triplets
    ES_transition_pair_count = get_transition_pairs(ES_tags_with_start_stop)
    ES_transition_pair_parameters = get_transition_pairs_using_MLE(
        ES_unique_tags, ES_transition_pair_count, ES_tags_with_start_stop
    )
    ES_transition_triplet_count = get_transition_triplets(ES_tags_with_start_stop)
    ES_transition_triplet_parameters = get_transition_triplets_using_MLE(
        ES_unique_tags, ES_transition_triplet_count, ES_tags_with_start_stop
    )

    RU_train_data = _preprocess_training_file1(RU_TRAIN_DATA_FILE)
    RU_emission_parameters = get_emission_using_MLE(RU_train_data)

    # Get RU tags
    RU_tags, RU_tags_with_start_stop, RU_train_words = _preprocess_training_file2(
        RU_TRAIN_DATA_FILE
    )
    RU_unique_tags = get_unique_elements(RU_tags)

    # Get RU words
    RU_test_words = _preprocess_test_file2(RU_TEST_DATA_FILE)
    RU_unique_words = get_unique_elements(RU_train_words)

    # Get RU transition pairs and triplets
    RU_transition_pair_count = get_transition_pairs(RU_tags_with_start_stop)
    RU_transition_pair_parameters = get_transition_pairs_using_MLE(
        RU_unique_tags, RU_transition_pair_count, RU_tags_with_start_stop
    )
    RU_transition_triplet_count = get_transition_triplets(RU_tags_with_start_stop)
    RU_transition_triplet_parameters = get_transition_triplets_using_MLE(
        RU_unique_tags, RU_transition_triplet_count, RU_tags_with_start_stop
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
            ES_transition_pair_parameters,
            ES_transition_triplet_parameters,
        )
        ES_generated_tag_list = generate_predictions_viterbi(
            word, ES_unique_tags, ES_transition_triplet_parameters
        )
        ES_predicted_tags_list.append(ES_generated_tag_list)

    write_to_predicted_file(
        ES_PART4_PREDICTED_DATA_FILE, ES_test_words, ES_predicted_tags_list
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
            RU_transition_pair_parameters,
            RU_transition_triplet_parameters,
        )
        RU_generated_tag_list = generate_predictions_viterbi(
            word, RU_unique_tags, RU_transition_triplet_parameters
        )
        RU_predicted_tags_list.append(RU_generated_tag_list)

    write_to_predicted_file(
        RU_PART4_PREDICTED_DATA_FILE, RU_test_words, RU_predicted_tags_list
    )

    print("\nResults for ES dataset")
    evaluateScores(ES_ACTUAL_DATA_FILE, ES_PART4_PREDICTED_DATA_FILE)

    print("\nResults for RU dataset")
    evaluateScores(RU_ACTUAL_DATA_FILE, RU_PART4_PREDICTED_DATA_FILE)
