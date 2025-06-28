#!/bin/bash
# Setup script for Bigfoot data preprocessing

echo "Creating virtual environment in current directory..."
python3 -m venv ./venv_bigfoot
source ./venv_bigfoot/bin/activate

echo "Installing Python libraries..."
# Ensure pip itself is up-to-date in the venv
./venv_bigfoot/bin/python -m pip install --upgrade pip
./venv_bigfoot/bin/pip install pandas numpy spacy nltk requests scikit-learn # Added scikit-learn for percentile, though numpy can do it.

echo "Downloading spaCy model..."
./venv_bigfoot/bin/python -m spacy download en_core_web_sm

echo "Downloading NLTK resources..."
./venv_bigfoot/bin/python -m nltk.downloader vader_lexicon punkt averaged_perceptron_tagger wordnet omw-1.4

echo "Setup complete. Activate the virtual environment using: source venv_bigfoot/bin/activate"
