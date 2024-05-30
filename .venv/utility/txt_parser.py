def extract_file_to_array(file_path):
    """
    Extract the contents of a text file into an array.

    :param file_path: Path to the text file
    :return: List containing the lines of the text file
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

def contains_txt_file(path):
    """
    Check if the path contains a text file.

    :param path: Path to check
    :return: True if the path contains a text file, False otherwise
    """
    return os.path.isfile(path) and path.lower().endswith('.txt')