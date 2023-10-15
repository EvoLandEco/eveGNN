import sys
import os


def main():
    # The base directory path is passed as the first argument
    name = sys.argv[1]

    print(f'Project: {name}')

    # The set_i folder names are passed as the remaining arguments
    set_paths = sys.argv[2:]

    for path in set_paths:
        # Concatenate the base directory path with the set_i folder name
        full_dir = os.path.join(name, path)
        # Call read_rds_to_pytorch with the full directory path
        print(full_dir)  # The set_i folder names are passed as the remaining arguments


if __name__ == '__main__':
    main()
