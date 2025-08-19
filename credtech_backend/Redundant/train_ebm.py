#!/usr/bin/env python3
"""
Script to train the EBM model using CSV data.
"""
import sys
import os

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.services.structured_service import StructuredModelService

def main():
    """Train the EBM model."""
    print("🚀 Starting EBM model training...")
    
    try:
        # Initialize the service
        service = StructuredModelService()
        
        # Train the model
        result = service.train()
        
        print("\n✅ Training completed successfully!")
        print(f"📊 Results:")
        print(f"   - Accuracy: {result['accuracy']:.4f}")
        print(f"   - Training samples: {result['training_samples']}")
        print(f"   - Test samples: {result['test_samples']}")
        print(f"   - Features: {result['feature_count']}")
        print(f"   - Model type: {result['model_type']}")
        print(f"   - Model saved to: {result['model_path']}")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
