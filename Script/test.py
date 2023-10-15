import sys


def main():
    # The base directory path is passed as the first argument
    name = sys.argv[1]

    print(f'Project: {name}')

    # The set_i folder names are passed as the remaining arguments
    set_paths = sys.argv[2:]

    for i, arg in enumerate(sys.argv[2:], start=1):
        print(f'Argument {i}: {arg}')


if __name__ == '__main__':
    main()
