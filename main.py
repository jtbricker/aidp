#!/usr/bin/python

import logging
import argparse
from aidp.data.modeldata import ModelData

def main():
    """ Parses the command line arguments and determines what to do """
    # Parse arguments
    args = parse_arguments()

    # Configure and get logger
    configure_logger(args.verbose)
    logger = logging.getLogger(__name__)
    logger.info("Starting AIDP Application")

    # Read in csv input file
    in_data = ModelData(args.input_file).read_data()
    logger.debug(in_data.data.head())

    # Read model file

    # Predict Results

    # Format Results

    # Write csv output

    logger.info("Ending AIDP Application")


def configure_logger(verbose):
    """ Configures logger used throughout the application
        To get an instance of the configured logger from any module, simply call:
            `logger = logging.getLogger(__name__)`

        Arguments:
            verbose {bool} -- Flag for indicating whether verbose mode is on
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler("out.log"),
            logging.StreamHandler()
        ]
    )


def parse_arguments():
    """Reads in arguments from command line

    Returns:
        dictionary -- dictionary of arguments passed from the command line
    """
    parser = argparse.ArgumentParser("aidp")
    parser.add_argument(
        "input_file", help="Input excel file with data you'd like to get predictions for")
    parser.add_argument(
        "output_file", help="Name of the excel file where you want the output to be saved")
    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                action="store_true")
    return parser.parse_args()
