#!/usr/bin/python

import logging
import argparse
from datetime import datetime

from aidp.data.modeldata import ModelData
from aidp.data.reader import ExcelDataReader
from aidp.runners.engines import getEngine


def main():
    """ Parses the command line arguments and determines what to do """
    # Parse arguments
    args = parse_arguments()

    # Configure and get logger
    configure_logger(args.verbose)
    logger = logging.getLogger(__name__)
    logger.info("Starting AIDP Application")

    # Read in csv input file
    model_data = ModelData(args.input_file, ExcelDataReader()).read_data()
    logger.debug(model_data.data.describe())

    # Get prediction/training engine
    engine = getEngine(args.cmd, model_data)
    engine.start(model_key=args.model_key)

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

    subparser = parser.add_subparsers(dest='cmd')
    subparser.required = True

    parser_predict = subparser.add_parser("predict")
    parser_predict.add_argument(
        "input_file", help="Input excel file with data you'd like to get predictions for")
    parser_predict.add_argument("-v", "--verbose", help="increase output verbosity",
                        action="store_true")
    parser_predict.add_argument("--model_key", help="""(Optional) Name of the models to use for predictions.
    These models should exist in their own folder in /resources/models/<model_key>.  If no model is
     provided 'default' is used""", default='default')

    parser_train = subparser.add_parser("train")
    parser_train.add_argument(
        "input_file", help="Input excel file with data you'd like to train the models on")
    parser_train.add_argument("-v", "--verbose", help="increase output verbosity",
                        action="store_true")
    parser_train.add_argument("--model_key", help="""(Optional) Name of the folder where the newly 
    trained will be stored (in /resources/models/<model_key>).  If no name is provided, a timestamp
    is used""", default=datetime.now().strftime("%Y-%m-%d-%H%M%S%f"))

    return parser.parse_args()

if __name__ == '__main__':
    main()
