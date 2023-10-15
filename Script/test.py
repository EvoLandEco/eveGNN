import sys


def main():
    # sys.argv[0] is the script name, so we start from sys.argv[1]
    for i, arg in enumerate(sys.argv[1:], start=1):
        print(f'Argument {i}: {arg}')


if __name__ == '__main__':
    main()
