#!/usr/bin/python

import logging
import argparse
from data.modeldata import ModelData

def main():
    # Parse arguments
    args = parse_arguments()

    # Configure and get logger
    configure_logger()
    logger = logging.getLogger(__name__)
    logger.info("Starting AIDP Application")

    # Read in csv input file
    in_data = ModelData(args.input_file).read_data()

    # Read model file


    # Predict Results

    # Format Results

    # Write csv output

    logger.info("Ending AIDP Application")

def configure_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler("out.log"),
            logging.StreamHandler()
        ]
    )

def parse_arguments():
    parser = argparse.ArgumentParser("aidp")
    parser.add_argument("input_file", help="Excel file with the same columns and column ordering as the original excel files used to train the model")
    parser.add_argument("output_file", help="Name of the excel file where you want the output to be saved")
    return parser.parse_args()
