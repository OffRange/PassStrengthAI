from urllib.request import urlopen
import pandas as pd

from config import CONFIG

lines = urlopen(CONFIG['cracked_passwords_url'])
lines = [line.decode('utf-8').replace('\\n', '') for line in lines.readlines()]


def __get_cracked_passwords(max_length: int):
    return [line for line in lines if len(line) <= max_length]


def __load_csv_passwords(csv):
    data = pd.read_csv(csv)
    data = data.sample(frac=1)

    return data.loc[data['strength'] == 0]['password'].tolist(), \
        data.loc[data['strength'] == 1]['password'].tolist(), \
        data.loc[data['strength'] == 2]['password'].tolist(), \
        data.loc[data['strength'] == 3]['password'].tolist(), \
        data.loc[data['strength'] == 4]['password'].tolist()


poor, weak, moderate, ok, strong = __load_csv_passwords(CONFIG['csv_password_dataset'])

LABEL_LOOKUP = {
    0: "Cracked",
    1: "Ridiculous",
    2: "Weak",
    3: "Moderate",
    4: "Strong",
    5: "Very strong"
}


def get_dataset(max_length, test_ratio=.3):
    print(f'Dataset with {max_length}-character passwords and {test_ratio} test ratio is loaded')
    min_dataset_length = min(len(weak), len(moderate), len(ok), len(strong))
    ratio = int(min_dataset_length * test_ratio)

    cracked_passwords = __get_cracked_passwords(max_length)[:min_dataset_length]
    poor_passwords = poor
    weak_passwords = weak[:min_dataset_length]
    moderate_passwords = moderate[:min_dataset_length]
    ok_passwords = ok[:min_dataset_length]
    strong_passwords = strong[:min_dataset_length]

    train_passwords = []
    train_passwords.extend(cracked_passwords[ratio:])
    train_passwords.extend(poor_passwords)
    train_passwords.extend(weak_passwords[ratio:])
    train_passwords.extend(moderate_passwords[ratio:])
    train_passwords.extend(ok_passwords[ratio:])
    train_passwords.extend(strong_passwords[ratio:])

    train_labels = [0 for _ in range(min_dataset_length - ratio)]
    train_labels.extend([1 for _ in range(len(poor_passwords))])
    train_labels.extend([i for i in range(2, 6) for _ in range(min_dataset_length - ratio)])

    test_passwords = []
    test_passwords.extend(cracked_passwords[:ratio])
    test_passwords.extend(weak_passwords[:ratio])
    test_passwords.extend(moderate_passwords[:ratio])
    test_passwords.extend(ok_passwords[:ratio])
    test_passwords.extend(strong_passwords[:ratio])
    test_labels = [0 for _ in range(ratio)]
    test_labels.extend([i for i in range(2, 6) for _ in range(ratio)])

    return Dataset((train_passwords, train_labels), (test_passwords, test_labels))


class Dataset:
    def __init__(self, train_dataset, test_dataset):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.total = len(self.train_dataset[0])

    def print_stats(self):
        length = len(max(LABEL_LOOKUP.values(), key=len))
        for index in LABEL_LOOKUP:
            print(f'{LABEL_LOOKUP[index]} {" " * (length - len(LABEL_LOOKUP[index]))}: {self.train_dataset[1].count(index):,}')

        print(f'Total {" " * (length - 5)}: {self.total:,}')
