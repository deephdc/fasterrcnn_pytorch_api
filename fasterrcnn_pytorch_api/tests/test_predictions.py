#import json, io
 
#def test_prediction(test_predict):
#    """Test the predict function."""
#    # Access the test_predict fixture defined in conftest.py
#    result, accept = test_predict

    # Assert the expected result based on the 'accept' argument
#    if accept == 'image/png':
#        assert isinstance(result, io.BytesIO)  # Ensure the result is a binary image
#    else:
#        assert isinstance(result, str)  
#        try:
#            json.loads(result) 
#        except json.JSONDecodeError:
#            assert False, "Result is not a valid JSON file"


