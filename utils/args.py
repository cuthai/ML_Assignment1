import argparse


def args():
    """
    Function to create command line arguments
    """
    # Initialize the parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument('-dn', '--data_name', help='Specify data name to extract and process')

    # Parse arguments
    command_args = parser.parse_args()

    # Return the parsed arguments
    return command_args
