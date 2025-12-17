"""
Run Evaluation Script - Hybrid Solution
"""

import argparse
from pathlib import Path

from src.recommender import SHLRecommender
from src.evaluator import evaluate_on_train_set, generate_test_predictions, print_evaluation_report
from src.scraper import scrape_shl_catalog, validate_scraped_data


def main():
    parser = argparse.ArgumentParser(description='Evaluate SHL Recommendation Engine')
    parser.add_argument('--data', type=str, required=True, help='Path to Excel file')
    parser.add_argument('--output', type=str, default='test_predictions.csv', help='Output CSV file')
    parser.add_argument('--k', type=int, default=10, help='K value for Recall@K')
    parser.add_argument('--predict-only', action='store_true', help='Only generate test predictions')
    parser.add_argument('--show-scraping', action='store_true', help='Show web scraping pipeline')
    parser.add_argument('--no-embeddings', action='store_true', help='Disable embeddings (TF-IDF only)')
    
    args = parser.parse_args()
    
    if not Path(args.data).exists():
        print(f"❌ Error: Data file not found: {args.data}")
        return
    
    print("=" * 70)
    print("SHL ASSESSMENT RECOMMENDATION ENGINE")
    print("=" * 70)
    
    # Show scraping pipeline if requested
    if args.show_scraping:
        assessments = scrape_shl_catalog()
        validate_scraped_data(assessments)
        print()
    
    # Initialize recommender
    print("\nInitializing recommender...")
    use_embeddings = not args.no_embeddings
    recommender = SHLRecommender(use_embeddings=use_embeddings)
    print(f"Loaded {recommender.num_assessments} assessments")
    
    if args.predict_only:
        # Generate test predictions only
        print("\n" + "=" * 70)
        print("GENERATING TEST PREDICTIONS")
        print("=" * 70)
        
        try:
            df = generate_test_predictions(
                excel_path=args.data,
                output_path=args.output,
                recommender=recommender,
                k=args.k
            )
            
            print(f"\n✅ Test predictions saved to: {args.output}")
            print(f"   Total predictions: {len(df)}")
            print(f"   Format: query, assessment_url")
        except Exception as e:
            print(f"❌ Error generating predictions: {e}")
            import traceback
            traceback.print_exc()
    else:
        # Full evaluation on train set
        print("\n" + "=" * 70)
        print("EVALUATING ON TRAINING SET")
        print("=" * 70)
        
        try:
            mean_recall, results = evaluate_on_train_set(
                train_file=args.data,
                recommender=recommender,
                k=args.k,
                verbose=True
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
                output_path=args.output,
                recommender=recommender,
                k=args.k
            )
            
            print(f"\n✅ Test predictions saved to: {args.output}")
            print(f"   Total predictions: {len(df)}")
        except Exception as e:
            print(f"❌ Error generating predictions: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
