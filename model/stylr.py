import importlib
from utils.base import parse_args


def main():
    args = parse_args()
    model = importlib.import_module(args.model)
    
    model.main(args)


if __name__ == '__main__':
    main()