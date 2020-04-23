import os
import inspect
import re
from tqdm import tqdm
from transformers import GPT2Tokenizer
from experiment import Model, Intervention
import pandas as pd

# Stats from https://arxiv.org/pdf/1804.06876.pdf, Table 1
OCCUPATION_FEMALE_PCT = {
    'carpenter': 2,
    'mechanic': 4,
    'construction worker': 4,
    'laborer': 4,
    'driver': 6,
    'sheriff': 14,
    'mover': 18,
    'developer': 20,
    'farmer': 22,
    'guard': 22,
    'chief': 27,
    'janitor': 34,
    'lawyer': 35,
    'cook': 38,
    'physician': 38,
    'ceo': 39,
    'analyst': 41,
    'manager': 43,
    'supervisor': 44,
    'salesperson': 48,
    'editor': 52,
    'designer': 54,
    'accountant': 61,
    'auditor': 61,
    'writer': 63,
    'baker': 65,
    'clerk': 72,
    'cashier': 73,
    'counselor': 73,
    'attendant': 76,
    'teacher': 78,
    'tailor': 80,
    'librarian': 84,
    'assistant': 85,
    'cleaner': 89,
    'housekeeper': 89,
    'nurse': 90,
    'receptionist': 90,
    'hairdresser': 92,
    'secretary': 95
}

def load_dev_examples(path='winobias_data/', subjective_filter=False, verbose=False):
    return load_examples(path, 'dev', subjective_filter, verbose)

def load_test_examples(path='winobias_data/', subjective_filter=False, verbose=False):
    return load_examples(path, 'test', subjective_filter, verbose)

def load_examples(path, split, subjective_filter=False, verbose=False):
    print(f'Split: {split.upper()}, Subjective Filter: {subjective_filter}')
    with open(os.path.join(path, 'female_occupations.txt')) as f:
        female_occupations = [row.lower().strip() for row in f]
    with open(os.path.join(path, 'male_occupations.txt')) as f:
        male_occupations = [row.lower().strip() for row in f]
    occupations = female_occupations + male_occupations

    if subjective_filter:
        fname = f'pro_stereotyped_type1.txt.{split}.subjectivefilter'
    else:
        fname = f'pro_stereotyped_type1.txt.{split}'

    with open(os.path.join(path, fname)) as f:
        examples = []
        row_pair = []
        skip_count = 0
        for row in f:
            row_pair.append(row)
            if len(row_pair) == 2:
                skip = False
                if row_pair[0].count('[') != 2 or row_pair[1].count('[') != 2: # Multiple pronouns
                    skip = True
                elif '[him]' in row_pair[0] + row_pair[1]: # Objective pronoun, almost always at end of sentence
                    skip = True
                else:
                    base_string1, substitutes1, continuation1, occupation1 = _parse_row(row_pair[0], occupations)
                    base_string2, substitutes2, continuation2, occupation2 = _parse_row(row_pair[1], occupations)
                    if base_string1 != base_string2 or substitutes1 != substitutes2:
                        skip = True
                if skip:
                    if verbose:
                        print('Skipping: ', row_pair)
                    skip_count += 1
                    row_pair = []
                    continue
                base_string = base_string1
                assert substitutes1 == substitutes2
                female_pronoun, male_pronoun = substitutes1
                assert len(continuation1) > 0 and len(continuation2) > 0 and continuation1 != continuation2
                assert len(occupation1) > 0 and len(occupation2) > 0 and occupation1 != occupation2
                if occupation1 in female_occupations:
                    female_occupation = occupation1
                    female_occupation_continuation = continuation1
                    male_occupation = occupation2
                    male_occupation_continuation = continuation2
                    assert occupation2 in male_occupations
                else:
                    male_occupation = occupation1
                    male_occupation_continuation = continuation1
                    female_occupation = occupation2
                    female_occupation_continuation = continuation2
                    assert occupation1 in male_occupations
                    assert occupation2 in female_occupations
                examples.append(WinobiasExample(base_string, female_pronoun, male_pronoun, female_occupation, male_occupation,
                 female_occupation_continuation, male_occupation_continuation))
                row_pair = []
        assert row_pair == []
    print(f'Loaded {len(examples)} pairs. Skipped {skip_count} pairs.')
    return examples


def _parse_row(row, occupations):
    _, sentence = row.strip().split(' ', 1)
    occupation = None
    for occ in occupations:
        if f'[the {occ.lower()}]' in sentence.lower():
            assert occupation is None
            occupation = occ.lower()
    assert occupation is not None

    pronoun_groups = [ # First element is female, second is male
        ('she', 'he'), # nominative
        ('her', 'his') # possessive
    ]

    num_matches = 0
    substitutes = None
    for pronouns in pronoun_groups:
        pattern = '|'.join(r'\[' + p + r'\]' for p in pronouns) # matches '[he]', '[she]', etc.
        pronoun_matches = re.findall(pattern, sentence)
        assert len(pronoun_matches) <= 1
        if pronoun_matches:
            num_matches += 1
            pronoun_match = pronoun_matches[0]
            context, continuation = sentence.split(pronoun_match)
            context = context.replace('[', '').replace(']', '')
            context = context.strip()
            assert '[' not in continuation
            continuation = continuation.strip()
            substitutes = pronouns
    assert num_matches == 1
    base_string = context + ' {}'

    return base_string, substitutes, continuation, occupation


def _odds_ratio(model, female_context, male_context, candidates):
    prob_female_occupation_continuation_given_female_pronoun, prob_male_occupation_continuation_given_female_pronoun = \
        model.get_probabilities_for_examples_multitoken(female_context, candidates)
    prob_female_occupation_continuation_given_male_pronoun, prob_male_occupation_continuation_given_male_pronoun = \
        model.get_probabilities_for_examples_multitoken(male_context, candidates)

    odds_given_female_pronoun = prob_female_occupation_continuation_given_female_pronoun / \
                                prob_male_occupation_continuation_given_female_pronoun
    odds_given_male_pronoun = prob_female_occupation_continuation_given_male_pronoun / \
                              prob_male_occupation_continuation_given_male_pronoun
    return odds_given_female_pronoun / odds_given_male_pronoun


def analyze(examples, gpt2_version='gpt2'):
    tokenizer = GPT2Tokenizer.from_pretrained(gpt2_version)

    model = Model(gpt2_version=gpt2_version)
    data = []
    for ex in tqdm(examples):
        candidates = [ex.female_occupation_continuation, ex.male_occupation_continuation]
        substitutes = [ex.female_pronoun, ex.male_pronoun]
        intervention = Intervention(tokenizer, ex.base_string, substitutes, candidates)
        female_context = intervention.base_strings_tok[0]
        male_context = intervention.base_strings_tok[1]
        odds_ratio = _odds_ratio(model, female_context, male_context, intervention.candidates_tok)
        female_pronoun = female_context[-1:]
        male_pronoun = male_context[-1:]
        odds_ratio_no_context = _odds_ratio(model, female_pronoun, male_pronoun, intervention.candidates_tok)
        desc = f'{ex.base_string.replace("{}", ex.female_pronoun + "/" + ex.male_pronoun)} // {ex.female_occupation_continuation} // {ex.male_occupation_continuation}'
        female_occupation_female_pct = OCCUPATION_FEMALE_PCT[ex.female_occupation]
        male_occupation_female_pct = OCCUPATION_FEMALE_PCT[ex.male_occupation]

        data.append({'odds_ratio': odds_ratio,
                     'odds_ratio_no_context': odds_ratio_no_context,
                     'female_occupation': ex.female_occupation,
                     'male_occupation': ex.male_occupation,
                     'desc': desc,
                     'occupation_pct_ratio': female_occupation_female_pct / male_occupation_female_pct})
    return pd.DataFrame(data)


class WinobiasExample():

    def __init__(self, base_string, female_pronoun, male_pronoun, female_occupation, male_occupation,
                 female_occupation_continuation, male_occupation_continuation):
        self.base_string = base_string
        self.female_pronoun = female_pronoun
        self.male_pronoun = male_pronoun
        self.female_occupation = female_occupation
        self.male_occupation = male_occupation
        self.female_occupation_continuation = female_occupation_continuation
        self.male_occupation_continuation = male_occupation_continuation

    def to_intervention(self, tokenizer):
        return Intervention(
            tokenizer=tokenizer,
            base_string=self.base_string,
            substitutes=[self.female_pronoun, self.male_pronoun],
            candidates=[self.female_occupation_continuation, self.male_occupation_continuation]
        )

    def __str__(self):
        return inspect.cleandoc(f"""
            base_string: {self.base_string}
            female_pronoun: {self.female_pronoun}
            male_pronoun: {self.male_pronoun}
            female_occupation: {self.female_occupation}
            male_occupation: {self.male_occupation}
            female_occupation_continuation: {self.female_occupation_continuation}
            male_occupation_continuation: {self.male_occupation_continuation}
        """)

    def __repr__(self):
        return str(self).replace('\n', ' ')


if __name__ == "__main__":
    load_test_examples(verbose=True)
