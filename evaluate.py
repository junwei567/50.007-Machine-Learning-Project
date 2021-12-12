from EvalScript.evalResult import get_observed, get_predicted,compare_observed_to_predicted

def evaluateScores(actual_file, predicted_file):
    with open(predicted_file, encoding="utf8") as f:
        predicted = f.read().splitlines()

    with open(actual_file, encoding="utf8") as f:
        actual = f.read().splitlines()
   
    compare_observed_to_predicted(get_observed(actual), get_predicted(predicted))

def write_to_predicted_file(predicted_file, words_list, tags_list):
    assert len(words_list) == len(tags_list)

    with open(predicted_file, "w", encoding="utf8") as f:
        for words, tags in zip(words_list, tags_list):  # Unpack all sentences and list of tags
            assert len(words) == len(tags)
            for word, tag in zip (words, tags):  # Unpack all words and tags
                f.write(f"{word} {tag}\n")
            f.write("\n")
