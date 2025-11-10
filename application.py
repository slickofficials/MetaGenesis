from flask import Flask, jsonify, Response
import logging
from prometheus_client import Counter, Histogram, generate_latest
import argparse
import os

# pylint: disable=line-too-long,logging-too-complex
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CounterService:
    """A simple microservice class to handle counter incrementation equivalent to the COBOL loop."""

    def __init__(self):
        """Initialize the class with counter set to 0."""
        self.counter = 0
        logger.info("CounterService initialized.")

    def increment_counter(self):
        """Perform the equivalent of PERFORM VARYING I FROM 1 UNTIL I > 10 ADD 1 TO COUNTER."""
        try:
            for i in range(1, 11):
                self.counter += 1
                logger.debug(f"Incremented counter to {self.counter} at iteration {i}")
            logger.info(f"Counter incremented 10 times, final value: {self.counter}")
            return self.counter
        except Exception as e:
            logger.error(f"Error during increment: {str(e)}")
            raise

# Metrics for pro observability
REQUEST_COUNT = Counter('counter_requests_total', 'Total requests')
INCREMENT_TIME = Histogram('counter_increment_seconds', 'Increment duration')

# Global instance for the service (in a real microservice, this could be per-request or singleton)
service = CounterService()

@app.route('/increment', methods=['GET'])
def api_increment():
    """API endpoint to trigger the counter increment."""
    REQUEST_COUNT.inc()  # Track every hit
    with INCREMENT_TIME.time():  # Time the op
        try:
            result = service.increment_counter()
            return jsonify({
                'status': 'success',
                'message': 'Counter incremented successfully.',
                'counter_value': result
            }), 200
        except Exception as e:
            logger.error(f"Error incrementing counter: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500

@app.route('/status', methods=['GET'])
def api_status():
    """API endpoint to check current counter value."""
    return jsonify({
        'status': 'ok',
        'counter_value': service.counter
    }), 200

@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint integrated into Flask."""
    return Response(generate_latest(), mimetype='text/plain')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Counter Microservice: API Server or CLI Test')
    parser.add_argument('--cli', action='store_true', help='Run CLI test mode (no server)')
    args = parser.parse_args()
    if args.cli:
        # Standalone CLI test—perfect for quick demos
        test_service = CounterService()
        print(f"CLI Mode: Initial value = {test_service.counter}")
        final_value = test_service.increment_counter()
        print(f"CLI Mode: After increment = {final_value}")
    else:
        # Run the API server (use $PORT for Beanstalk)
        port = int(os.getenv('PORT', 5000))
        app.run(host='0.0.0.0', port=port, debug=False)