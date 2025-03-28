import os
from google.cloud import aiplatform
import sys

def set_credentials():
    """Set up credentials for Windows environment"""
    creds_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "keys",
        "directed-reef-450323-t2-11d3aad6566c.json"
    )
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
    return creds_path

def test_authentication():
    """Test Google Cloud Authentication and Vertex AI Access"""
    print("Starting authentication test...\n")
    
    # 1. Set up credentials
    creds_path = set_credentials()
    print(f"Looking for credentials at: {creds_path}")
    
    if not os.path.exists(creds_path):
        print("❌ Credentials file not found at specified path!")
        print(f"Expected path: {creds_path}")
        print("Please ensure your credentials file is in the 'keys' folder.")
        return False
    print("✓ Credentials file found")
    
    try:
        # 2. Initialize Vertex AI with correct project ID
        print("\nTrying to initialize Vertex AI...")
        aiplatform.init(
            project='521902680111',  # Updated project ID
            location='us-central1'
        )
        print("✓ Successfully initialized Vertex AI")
        
        # 3. Access endpoint with correct endpoint ID
        print("\nTrying to access endpoint...")
        endpoint_name = "projects/521902680111/locations/us-central1/endpoints/8318633396381155328"
        endpoint = aiplatform.Endpoint(endpoint_name=endpoint_name)
        print("✓ Successfully accessed endpoint")
        
        # 4. Try a test prediction
        print("\nTrying a test prediction...")
        test_instance = {
            'Train_Number': '15014',
            'Source_Station': 'BNT',
            'Destination_Station': 'FZD'
        }
        
        try:
            response = endpoint.predict([test_instance])
            print("✓ Successfully made a test prediction")
            print(f"Test prediction result: {response.predictions[0]}")
            print("\n✅ All authentication tests passed successfully!")
            return True
        
        except Exception as prediction_error:
            print(f"\n❌ Error during prediction: {str(prediction_error)}")
            print("\nDebugging information:")
            print(f"Test instance format: {test_instance}")
            return False
            
    except Exception as e:
        print(f"\n❌ Error during authentication test: {str(e)}")
        return False

def list_available_endpoints():
    """List all available endpoints in the project"""
    try:
        print("\nListing available endpoints...")
        endpoints = aiplatform.Endpoint.list(
            project='521902680111',  # Updated project ID
            location='us-central1'
        )
        print("\nAvailable endpoints:")
        for endpoint in endpoints:
            print(f"- Name: {endpoint.display_name}")
            print(f"  Full path: {endpoint.resource_name}\n")
    except Exception as e:
        print(f"Error listing endpoints: {str(e)}")

if __name__ == "__main__":
    test_authentication()
    list_available_endpoints()