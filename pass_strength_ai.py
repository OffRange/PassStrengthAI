import config

from ai import try_model, train_model
from ai.model_v1 import Model
from dataset import get_dataset
from sys import argv


model_lookup = {
    'v1': Model
}


args = argv[1:]
if __name__ == '__main__':
    try:
        max_length = config.CONFIG['password_max_length']
    except KeyError:
        max_length = 15

    try:
        test_ratio = config.CONFIG['test_ratio']
    except KeyError:
        test_ratio = .3

    if "--max-length" in args:
        index = args.index("--max-length")
        if index + 1 < len(args):
            max_length = args[index + 1]

    verbose = "--verbose" in args
    try_model_input = '--try' in args

    if '--train' in args:
        index = args.index("--train")
        if index + 1 < len(args):
            model = args[index + 1]
            if model not in model_lookup:
                print(f'"{model}" is not valid. Choose one of: {", ".join(list(model_lookup.keys()))}')
                exit(1)

            dataset = get_dataset(max_length, test_ratio=test_ratio)
            if verbose:
                dataset.print_stats()

            model, loss, acc = train_model(model_lookup[model], dataset, max_length=max_length)

            model.save_model(filename=f'model_{dataset.total}-{int(loss*100)}-{int(acc*100)}.tflite')

            if try_model_input:
                try_model(model)

    if '--execute' in args:
        index = args.index("--execute")
        if index + 1 < len(args):
            model_path = args[index + 1]

            model = Model.load_from_file(model_path)
            if verbose:
                model.model.summary()

            try_model(model)