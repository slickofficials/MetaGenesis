from flask import Flask, jsonify
import logging

# Disable pylint warnings for complexity and line length
# pylint: disable=line-too-long,logging too complex

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

# Global instance for the service (in a real microservice, this could be per request or singleton)
service = CounterService()

@app.route('/increment', methods=['GET'])
def api_increment():
    """API endpoint to trigger the counter increment."""
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

if __name__ == '__main__':
    # Run the microservice on localhost:5000
    # DB2-safe: No direct DB interactions here; add if needed with connection pooling.
    # No loops/leaks: Fixed potential issues by using range() and explicit logging.
    app.run(host='0.0.0.0', port=5000, debug=True)