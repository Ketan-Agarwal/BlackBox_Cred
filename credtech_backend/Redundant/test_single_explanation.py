#!/usr/bin/env python3
"""Test script to generate explanation for a single company."""

import logging
from ebm_exp import EBMExplainer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Initialize explainer
    explainer = EBMExplainer(model_path='models/ebm_model_trained_on_csv.pkl')
    
    # Generate explanation for Service Corporation International (index 425, 0-based)
    csv_path = 'complete_training_dataset_corrected.csv'
    explainer.explain_multiple_samples(csv_path, sample_indices=[425], num_samples=1)
    
    print("Explanation regenerated for Service Corporation International")

if __name__ == "__main__":
    main()
