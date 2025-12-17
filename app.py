"""
SHL Assessment Recommendation Engine - Flask API
=================================================
Production-ready REST API for assessment recommendations.

Endpoints:
    GET  /health     - Health check
    POST /recommend  - Get assessment recommendations

Usage:
    python app.py
    
    Or with gunicorn:
    gunicorn app:app --bind 0.0.0.0:5000
"""

import os
import sys
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.recommender import SHLRecommender
from src.config import API_HOST, API_PORT, DEBUG_MODE, MAX_RECOMMENDATIONS, MIN_RECOMMENDATIONS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize recommender (singleton)
recommender = None


def get_recommender():
    """Get or create recommender instance."""
    global recommender
    if recommender is None:
        logger.info("Initializing recommendation engine...")
        recommender = SHLRecommender()
        logger.info(f"Loaded {recommender.num_assessments} assessments")
    return recommender


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health Check Endpoint
    
    Returns:
        JSON: {"status": "healthy"}
        
    Status Codes:
        200: Service is healthy
        500: Service is unhealthy
    """
    try:
        rec = get_recommender()
        if rec.num_assessments > 0:
            return jsonify({"status": "healthy"}), 200
        else:
            return jsonify({"status": "unhealthy", "reason": "No assessments loaded"}), 500
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({"status": "unhealthy", "reason": str(e)}), 500


@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Assessment Recommendation Endpoint
    
    Request Body:
        {
            "query": "JD/query in string"
        }
        
    Returns:
        JSON: {
            "recommended_assessments": [
                {
                    "url": "Valid URL in string",
                    "name": "Assessment name",
                    "adaptive_support": "Yes/No",
                    "description": "Description in string",
                    "duration": 60,
                    "remote_support": "Yes/No",
                    "test_type": ["List of string"]
                },
                ...
            ]
        }
        
    Status Codes:
        200: Success
        400: Bad request (missing or invalid query)
        500: Internal server error
    """
    try:
        # Parse request
        data = request.get_json()
        
        if not data:
            return jsonify({
                "error": "Request body must be JSON",
                "recommended_assessments": []
            }), 400
        
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({
                "error": "Query parameter is required",
                "recommended_assessments": []
            }), 400
        
        # Get number of results (optional parameter)
        num_results = data.get('num_results', MAX_RECOMMENDATIONS)
        num_results = max(MIN_RECOMMENDATIONS, min(MAX_RECOMMENDATIONS, int(num_results)))
        
        logger.info(f"Processing query: {query[:100]}...")
        
        # Get recommendations
        rec = get_recommender()
        response = rec.get_api_response(query, num_results)
        
        logger.info(f"Returning {len(response['recommended_assessments'])} recommendations")
        
        return jsonify(response), 200
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return jsonify({
            "error": f"Invalid request: {str(e)}",
            "recommended_assessments": []
        }), 400
        
    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        return jsonify({
            "error": "Internal server error",
            "recommended_assessments": []
        }), 500


@app.route('/', methods=['GET'])
def home():
    """Home page with API documentation."""
    return jsonify({
        "name": "SHL Assessment Recommendation Engine",
        "version": "1.0.0",
        "endpoints": {
            "GET /health": "Health check endpoint",
            "POST /recommend": "Get assessment recommendations",
        },
        "usage": {
            "recommend": {
                "method": "POST",
                "body": {"query": "Your job description or query"},
                "example": "curl -X POST -H 'Content-Type: application/json' -d '{\"query\": \"Java developer\"}' http://localhost:5000/recommend"
            }
        }
    }), 200


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Query Analysis Endpoint (for debugging)
    
    Returns detailed analysis of how the query is understood.
    """
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({"error": "Query parameter is required"}), 400
        
        rec = get_recommender()
        analysis = rec.analyze_query_details(query)
        
        return jsonify(analysis), 200
        
    except Exception as e:
        logger.error(f"Error analyzing query: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/assessments', methods=['GET'])
def list_assessments():
    """
    List All Assessments Endpoint (for debugging/admin)
    
    Returns all assessments in the catalog.
    """
    try:
        rec = get_recommender()
        assessments = rec.get_all_assessments()
        
        return jsonify({
            "total": len(assessments),
            "assessments": assessments
        }), 200
        
    except Exception as e:
        logger.error(f"Error listing assessments: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed"}), 405


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    # Initialize recommender on startup
    get_recommender()
    
    # Run the app
    port = int(os.environ.get('PORT', API_PORT))
    debug = os.environ.get('DEBUG', str(DEBUG_MODE)).lower() == 'true'
    
    logger.info(f"Starting server on {API_HOST}:{port}")
    app.run(host=API_HOST, port=port, debug=debug)
