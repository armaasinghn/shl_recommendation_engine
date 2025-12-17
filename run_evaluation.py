"""
Run Evaluation Script
=====================
Evaluates the recommendation engine and generates test predictions.

Usage:
    # Evaluate on training set
    python run_evaluation.py --data data/Gen_AI_Dataset.xlsx
    
    # Generate test predictions
    python run_evaluation.py --predict-only --data data/Gen_AI_Dataset.xlsx --output predictions.csv
"""

import argparse
from pathlib import Path

from src.recommender import SHLRecommender
from src.evaluator import (
    evaluate_on_train_set,
    generate_test_predictions,
    print_evaluation_report
)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate SHL Recommendation Engine'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to Excel file with Train-Set and Test-Set'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='test_predictions.csv',
        help='Output CSV file for test predictions'
    )
    
    parser.add_argument(
        '--k',
        type=int,
        default=10,
        help='K value for Recall@K metric'
    )
    
    parser.add_argument(
        '--predict-only',
        action='store_true',
        help='Only generate test predictions (skip training evaluation)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress'
    )
    
    args = parser.parse_args()
    
    # Validate files
    if not Path(args.data).exists():
        print(f"❌ Error: Data file not found: {args.data}")
        return
    
    print("=" * 70)
    print("SHL Assessment Recommendation Engine")
    print("=" * 70)
    
    # Initialize recommender
    print("\nInitializing recommender...")
    recommender = SHLRecommender(use_llm=True)
    print(f"Loaded {recommender.num_assessments} assessments")
    
    if args.predict_only:
        # Only generate test predictions
        print("\n" + "=" * 70)
        print("GENERATING TEST PREDICTIONS")
        print("=" * 70)
        
        try:
            df = generate_test_predictions(
                excel_path=args.data,
                output_path=args.output,  # <- FIXED: Changed from output_csv to output_path
                recommender=recommender,
                k=args.k,
                verbose=args.verbose
            )
            
            print(f"\n✅ Test predictions saved to: {args.output}")
            print(f"   Total predictions: {len(df)}")
            print(f"   Format: query, assessment_url")
            
        except Exception as e:
            print(f"❌ Error generating predictions: {e}")
            import traceback
            traceback.print_exc()
    
    else:
        # Evaluate on training set
        print("\n" + "=" * 70)
        print("EVALUATING ON TRAINING SET")
        print("=" * 70)
        
        try:
            mean_recall, results = evaluate_on_train_set(
                train_file=args.data,
                recommender=recommender,
                k=args.k,
                verbose=args.verbose
            )
            
            print_evaluation_report(results, k=args.k)
            
        except Exception as e:
            print(f"❌ Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
        
        # Generate test predictions
        print("\n" + "=" * 70)
        print("GENERATING TEST PREDICTIONS")
        print("=" * 70)
        
        try:
            df = generate_test_predictions(
                excel_path=args.data,
                output_path=args.output,  # <- FIXED: Changed from output_csv to output_path
                recommender=recommender,
                k=args.k,
                verbose=args.verbose
            )
            
            print(f"\n✅ Test predictions saved to: {args.output}")
            print(f"   Total predictions: {len(df)}")
            print(f"   Format: query, assessment_url")
            
        except Exception as e:
            print(f"❌ Error generating predictions: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()