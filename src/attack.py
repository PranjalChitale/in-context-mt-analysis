import argparse
import random
import string
from copy import deepcopy

import regex as re

indic_punctuation_pattern = re.compile(
    r"(["
    + string.punctuation
    + r"\u0964\u0965\uAAF1\uAAF0\uABEB\uABEC\uABED\uABEE\uABEF\u1C7E\u1C7F"
    + r"])"
)
urdu_punctuation_pattern = re.compile(
    r"(["
    + string.punctuation
    + r"\u0609\u060A\u060C\u061E\u066A\u066B\u066C\u066D\u06D4"
    + r"])"
)
latin_punctuation_pattern = re.compile("[" + string.punctuation + "]")

unicode_ranges = {
    "Latn": list(range(0x0020, 0x0080)),  # Basic Latin
    "Cyrl": list(range(0x0400, 0x052F)),  # Cyrillic
    "Deva": list(range(0x0900, 0x0980)),  # Devanagari
    "Beng": list(range(0x0980, 0x0A00)),  # Bengali
    "Guru": list(range(0x0A00, 0x0A80)),  # Gurmukhi
    "Gujr": list(range(0x0A80, 0x0B00)),  # Gujarati
    "Orya": list(range(0x0B00, 0x0B80)),  # Oriya
    "Taml": list(range(0x0B80, 0x0C00)),  # Tamil
    "Telu": list(range(0x0C00, 0x0C80)),  # Telugu
    "Knda": list(range(0x0C80, 0x0D00)),  # Kannada
    "Mlym": list(range(0x0D00, 0x0D80)),  # Malayalam
    "Sinh": list(range(0x0D80, 0x0DFF)),  # Sinhala
    "Olck": list(range(0x1C50, 0x1C7F)),  # Santali
    "Arab": list(range(0x0600, 0x06FF)),  # Urdu
}


def duplicate_word_attack(sentence: str, repetition_percentage: float):
    """Perturbs the text sentence by randomly duplicating words from the given sentence.

    Args:
        sentence (str): Input text sentence which will be modified.
        repetition_percentage (float): Fraction of words to be duplicated (between 0 and 1).

    Returns:
        str: Perturbed text sentence with randomly duplicated words.
    """
    words = sentence.split()
    if not words:
        return sentence

    # calculate the number of words to repeat and randomly choose words to repeat
    num_words_to_repeat = max(1, int(len(words) * repetition_percentage))
    words_to_repeat = random.sample(words, num_words_to_repeat)

    # duplicate a random occurrence of each chosen word
    for word in words_to_repeat:
        occurrences = [i for i, w in enumerate(words) if w == word]
        if occurrences:
            chosen_occurrence = random.choice(occurrences)
            words.insert(chosen_occurrence + 1, word)

    # reconstruct the sentence with duplicated words
    sentence = " ".join(words)
    return sentence


def add_span_noise(sentence: str, language: str, noise_percentage: float):
    """Perturbs the text string by randomly deleting or replacing spans of characters within
    the valid unicode range for the specified language and script of the given sentence.

    Args:
        sentence (str): Input text sentence which will be modified.
        language (str): String indicating the language (ISO 639-3) and script (ISO 15924)
            in the FLORES-200 format (e.g., eng_Latn).
        noise_percentage (float): Fraction of characters to be modified (between 0 and 1).

    Returns:
        str: Perturbed text sentence with specified span noise budget.
    """
    # get the unicode range for specified language and script
    lang, script = language.split("_")
    unicode_range = unicode_ranges.get(script, [])

    # explcitly remove punctuations as potential replacement choices as an
    # alternate punctuation perturbation strategy is defined already below
    if language.endswith("Arab"):
        unicode_range = [
            i
            for i in unicode_range
            if not re.match(urdu_punctuation_pattern, chr(i))
        ]
    elif language.endswith("Latn") or language.endswith("Cyrl"):
        unicode_range = [
            i
            for i in unicode_range
            if not re.match(latin_punctuation_pattern, chr(i))
        ]
    else:
        unicode_range = [
            i
            for i in unicode_range
            if not re.match(indic_punctuation_pattern, chr(i))
        ]

    # create a list of indices for the characters in the sentence within the specified unicode range
    indices = [
        i for i, char in enumerate(sentence) if ord(char) in unicode_range
    ]

    # calculate the noise injection budget
    noise_budget = (int)(noise_percentage * len(indices))

    while noise_budget > 0:
        indices = [
            i for i, char in enumerate(sentence) if ord(char) in unicode_range
        ]

        # 1. randomly select a subset of indices as the starting point for noise injection
        # 2. sample a random span length between 1 and 3 characters
        # 3. calculate the end index of the span
        start_index = random.choice(indices)
        span_length = min(random.randint(1, 3), noise_budget)
        end_index = min(start_index + span_length, len(sentence))

        # determine the type of noise operation (deletion or replacement)
        noise_operation = random.choice(["deletion", "replacement"])

        if noise_operation == "deletion":
            # delete the selected span
            sentence = sentence[:start_index] + "" + sentence[end_index:]
        else:
            # replace the selected span with a single random character from the language-specific unicode range
            replacement_char = chr(random.choice(unicode_range))
            sentence = (
                sentence[:start_index] + replacement_char + sentence[end_index:]
            )

        # update the noise budget and terminate if the noise budget is exhausted
        noise_budget -= span_length
        if noise_budget <= 0:
            break

    return sentence


def punctuation_drop_attack(
    sentence: str, language: str, punctuation_percentage: float
):
    """Perturbs the text string by randomly deleting or replacing spans of characters within
    the valid unicode range for the specified language and script of the given sentence.

    Args:
        sentence (str): Input text sentence which will be modified.
        language (str): String indicating the language (ISO 639-3) and script (ISO 15924)
            in the FLORES-200 format (e.g., eng_Latn).
        punctuation_percentage (float): Fraction of punctuation characters to be removed (between 0 and 1).

    Returns:
        str: Perturbed text sentence with dropped punctuation characters.
    """
    # get the language and script specific punctuation patterns
    if language.endswith("Arab"):
        punctuation_pattern = urdu_punctuation_pattern
    elif language.endswith("Latn") or language.endswith("Cyrl"):
        punctuation_pattern = latin_punctuation_pattern
    else:
        punctuation_pattern = indic_punctuation_pattern

    # find all punctuation characters using the language-specific pattern
    punctuation_matches = punctuation_pattern.finditer(sentence)

    # convert the potential matches into a list of indices
    punctuation_indices = [match.start() for match in punctuation_matches]

    # calculate the number of punctuation characters to remove
    num_punctuation_to_remove = max(
        1, int(len(punctuation_indices) * punctuation_percentage)
    )

    # randomly choose punctuation indices to remove if any or skip.
    if len(punctuation_indices) == 0:
        indices_to_remove = []
    else:
        indices_to_remove = random.sample(
            punctuation_indices, k=num_punctuation_to_remove
        )

    # remove the selected punctuation characters
    sentence = "".join(
        [char for i, char in enumerate(sentence) if i not in indices_to_remove]
    )
    return sentence


def punctuation_add_attack(
    sentence: str, language: str, punctuation_percentage: float
):
    """Perturbs the text string by randomly adding punctuations to a percentage of words.

    Args:
        sentence (str): Input text sentence which will be modified.
        language (str): String indicating the language (ISO 639-3) and script (ISO 15924)
            in the FLORES-200 format (e.g., eng_Latn).
        punctuation_percentage (float): Fraction of words to add punctuation character (between 0 and 1).

    Returns:
        str: Perturbed text sentence with added punctuation characters to a fraction of words.
    """
    words = sentence.split()

    # get the unicode range for specified language and script
    lang, script = language.split("_")
    unicode_range = unicode_ranges.get(script, [])

    # get the language and script specific punctuation patterns
    if language.endswith("Arab"):
        punctuation_pattern = urdu_punctuation_pattern
    elif language.endswith("Latn") or language.endswith("Cyrl"):
        punctuation_pattern = latin_punctuation_pattern
    else:
        punctuation_pattern = indic_punctuation_pattern

    # build a set of acceptable punctuations based on the specified language and script
    valid_punctuations = [
        chr(i) for i in unicode_range if re.match(punctuation_pattern, chr(i))
    ]

    # defaults to english punctuations if no language-specific punctuations
    if len(valid_punctuations) == 0:
        valid_punctuations = [i for i in string.punctuation]

    # identify words that do not have punctuation attached
    words_without_punctuation = [
        word for word in words if not punctuation_pattern.search(word)
    ]

    # calculate the number of words to add punctuation to
    num_words_to_perturb = max(
        1, int(len(words_without_punctuation) * punctuation_percentage)
    )

    # randomly choose words without punctuation to perturb
    words_to_perturb = random.sample(
        words_without_punctuation, k=num_words_to_perturb
    )

    # concatenate random punctuations at the end of the chosen words
    for i in range(len(words)):
        if words[i] in words_to_perturb:
            words[i] += random.choice(valid_punctuations)

    # reconstruct the perturbed sentence with added punctuations
    sentence = " ".join(words)
    return sentence


def ocr_attack(sentence: str, ocr_percentage: float):
    """Perturbs the text string by randomly fusing or splitting words given a specific budget.

    Args:
        sentence (str): Input text sentence which will be modified.
        ocr_percentage (float): Fraction of words be modified (between 0 and 1).

    Returns:
        str: Perturbed text sentence after the OCR-based noise injection.
    """
    words = sentence.split()

    # calculate the OCR injection budget
    ocr_budget = (int)(ocr_percentage * len(words))

    # randomly select a subset of words as the starting point for OCR injection
    num_words_to_attack = max(1, int(ocr_percentage * len(words)))
    words_to_attack = random.sample(words, k=num_words_to_attack)

    # iterate over the selected words to perform OCR injection
    for i in range(len(words_to_attack)):
        start_word = words_to_attack[i]
        index = words.index(start_word)

        # randomly decide between fusion and split
        if index == len(words) - 1:
            ocr_operation = "split"
        else:
            ocr_operation = random.choice(["fusion", "split"])

        if ocr_operation == "fusion":
            next_index = index + 1
            if next_index < len(words):
                # concatenate the selected word with the next word.
                concatenated_word = start_word + words[next_index]
                # update words_to_attack to have the concatenation instead of next word
                words[index] = concatenated_word

                # if words_to_attack has the word being fused as a target to attack, replace it.
                if words[next_index] in words_to_attack:
                    occurences = [
                        idx
                        for idx in range(0, len(words_to_attack))
                        if words_to_attack[idx] == words[next_index]
                    ]
                    for idx in occurences:
                        if idx > i:
                            words_to_attack[idx] = concatenated_word

                # if current word is in words to attack change it.
                if start_word in words_to_attack:
                    occurences = [
                        idx
                        for idx in range(0, len(words_to_attack))
                        if words_to_attack[idx] == start_word
                    ]
                    for idx in occurences:
                        if idx > i:
                            words_to_attack[idx] = concatenated_word

                # remove the next word
                words.pop(next_index)
        else:
            # randomly determine the split point
            try:
                split_point = random.randint(1, len(start_word) - 1)
            except:  # noqa: E722
                continue

            # split the word into two and insert the second part at the next index
            words[index] = start_word[:split_point]
            words.insert(index + 1, start_word[split_point:])

            if start_word in words_to_attack:
                occurences = [
                    idx
                    for idx in range(0, len(words_to_attack))
                    if words_to_attack[idx] == start_word
                ]
                for idx in occurences:
                    if idx > i:
                        words_to_attack[idx] = random.choice(
                            [start_word[:split_point], start_word[split_point:]]
                        )

        # update the OCR budget and terminate if the OCR budget is exhausted
        ocr_budget -= 1
        if ocr_budget <= 0:
            break

    # reconstruct the sentence with manipulated words
    sentence = " ".join(words)
    return sentence


def random_word_ordering_attack(sentence: str, permutation_percentage: float):
    """Perturbs text sentence by randomly permuting a fraction of words in a given sentence.

    Args:
        sentence (str): Input text sentence which will be modified.
        permutation_percentage (float): Fraction of words be permuted (between 0 and 1).

    Returns:
        str: Perturbed text sentence with randomly permuted words.
    """
    words = sentence.split()
    if not words:
        return sentence

    # calculate the number of words to permute
    num_words_to_permute = max(1, int(len(words) * permutation_percentage))

    # get the indices of words to permute and shuffle them
    indices_to_permute = random.sample(range(len(words)), num_words_to_permute)
    indices_to_permute_ = deepcopy(indices_to_permute)
    random.shuffle(indices_to_permute_)

    # make sure that the indices are actually shuffled
    if indices_to_permute_ == indices_to_permute:
        indices_to_permute_ = sorted(indices_to_permute_)
        if indices_to_permute_ == indices_to_permute:
            indices_to_permute_ = sorted(indices_to_permute_, reverse=True)

    # permute the words based on shuffled indices
    perturbed_words = []
    for k in range(0, len(words)):
        if k in indices_to_permute:
            old_index = indices_to_permute.index(k)
            perturbed_words.append(words[indices_to_permute_[old_index]])
        else:
            perturbed_words.append(words[k])

    # reconstruct the sentence with permuted word order
    sentence = " ".join(perturbed_words)
    return sentence


def apply_attack(
    sentence: str, language: str, attack_type: str, **attack_params
):
    """Apply a specified attack to a given sentence.

    Args:
        sentence (str): Input text sentence which will be modified.
        language (str): String indicating the language (ISO 639-3) and script (ISO 15924)
            in the FLORES-200 format (e.g., eng_Latn).
        attack_type (str): The type of attack to be applied. Options include 'span_noise',
            'word_duplication', 'punctuation_drop_attack', 'punctuation_add_attack',
            'ocr', and 'word_order'.

    Raises:
        NotImplementedError: If an invalid attack type is provided.

    Returns:
        str: Pertubed text sentence after applying the specified attack.
    """
    if attack_type == "span_noise":
        return add_span_noise(sentence, language, **attack_params)
    elif attack_type == "word_duplication":
        return duplicate_word_attack(sentence, **attack_params)
    elif attack_type == "punctuation_drop_attack":
        return punctuation_drop_attack(sentence, language, **attack_params)
    elif attack_type == "punctuation_add_attack":
        return punctuation_add_attack(sentence, language, **attack_params)
    elif attack_type == "ocr":
        return ocr_attack(sentence, **attack_params)
    elif attack_type == "word_order":
        return random_word_ordering_attack(sentence, **attack_params)
    else:
        raise NotImplementedError("Invalid attack type. Choose a valid attack.")


def perturb_file(
    input_filename: str,
    output_filename: str,
    language: str,
    attack_type: str,
    **attack_params
):
    """Perturbs sentences from an input file using specified attack parameters and
    writes the perturbed sentences to an output file.

    Args:
        input_filename (str): Path of the input file containing sentences to be perturbed.
        output_filename (str): Path of the output file where perturbed sentences will be saved.
        language (str): String indicating the language (ISO 639-3) and script (ISO 15924)
            in the FLORES-200 format (e.g., eng_Latn).
        attack_type (str): The type of attack to be applied. Options include 'span_noise',
            'word_duplication', 'punctuation_drop_attack', 'punctuation_add_attack',
            'ocr', and 'word_order'.
    """
    with open(input_filename, "r", encoding="utf-8") as infile:
        sentences = infile.readlines()

    perturbed_sentences = []

    for sentence in sentences:
        perturbed_sentence = apply_attack(
            sentence.strip(), language, attack_type, **attack_params
        )
        perturbed_sentences.append(perturbed_sentence.strip().replace("\n", ""))

    with open(output_filename, "w", encoding="utf-8") as outfile:
        outfile.write("\n".join(perturbed_sentences))


def main():
    parser = argparse.ArgumentParser(
        description="Add span noise to a sentence."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        help="Input file containing sentences to perturb.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Output file to save perturbed sentences.",
    )
    parser.add_argument(
        "--language",
        type=str,
        help="Language of the input sentence in the format 'lang_Script'.",
    )
    parser.add_argument(
        "--attack_type",
        type=str,
        help="Choose an attack to apply on the input text.",
    )
    parser.add_argument(
        "--noise_percentage",
        type=float,
        default=0.1,
        help="Percentage of characters to inject noise into (default: 0.1).",
    )
    parser.add_argument(
        "--repetition_percentage",
        type=float,
        default=0.20,
        help="Percentage of words to inject repetition noise into (default: 0.1).",
    )
    parser.add_argument(
        "--punctuation_percentage",
        type=float,
        default=1.0,
        help="Percentage of punctuations to drop.",
    )
    parser.add_argument(
        "--ocr_percentage",
        type=float,
        default=0.20,
        help="Percentage of punctuations to inject OCR errors.",
    )
    parser.add_argument(
        "--permutation_percentage",
        type=float,
        default=0.2,
        help="Percentage of words to permute (default: 0.1).",
    )
    args = parser.parse_args()

    if args.attack_type == "span_noise":
        attack_params = {
            "noise_percentage": args.noise_percentage,
        }
        perturb_file(
            args.input_file,
            args.output_file,
            args.language,
            args.attack_type,
            **attack_params
        )
    elif args.attack_type == "word_duplication":
        attack_params = {
            "repetition_percentage": args.repetition_percentage,
        }
        perturb_file(
            args.input_file,
            args.output_file,
            args.language,
            args.attack_type,
            **attack_params
        )
    elif args.attack_type == "punctuation_drop_attack":
        attack_params = {
            "punctuation_percentage": args.punctuation_percentage,
        }
        perturb_file(
            args.input_file,
            args.output_file,
            args.language,
            args.attack_type,
            **attack_params
        )
    elif args.attack_type == "punctuation_add_attack":
        attack_params = {
            "punctuation_percentage": args.punctuation_percentage,
        }
        perturb_file(
            args.input_file,
            args.output_file,
            args.language,
            args.attack_type,
            **attack_params
        )
    elif args.attack_type == "ocr":
        attack_params = {
            "ocr_percentage": args.ocr_percentage,
        }
        perturb_file(
            args.input_file,
            args.output_file,
            args.language,
            args.attack_type,
            **attack_params
        )
    elif args.attack_type == "word_order":
        attack_params = {"permutation_percentage": args.permutation_percentage}
        perturb_file(
            args.input_file,
            args.output_file,
            args.language,
            args.attack_type,
            **attack_params
        )
    else:
        raise NotImplementedError(
            "Invalid attack type. Choose 'span_noise' or 'word_duplication'."
        )


if __name__ == "__main__":
    main()
