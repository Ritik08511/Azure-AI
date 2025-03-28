from google.cloud import aiplatform
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TrainDelayPredictor:
    def __init__(self, project_id="521902680111", endpoint_id="8318633396381155328", location="us-central1", credentials_path=None):
        logger.debug("Initializing TrainDelayPredictor...")
        logger.debug(f"Project ID: {project_id}")
        logger.debug(f"Endpoint ID: {endpoint_id}")
        logger.debug(f"Location: {location}")
        
        try:
            if credentials_path:
                aiplatform.init(
                    project=project_id,
                    location=location,
                    credentials=aiplatform.Credentials.from_service_account_file(credentials_path)
                )
            self.endpoint = aiplatform.Endpoint(
                endpoint_name=f"projects/{project_id}/locations/{location}/endpoints/{endpoint_id}"
            )
            self.is_available = True
        except Exception as e:
            logger.error(f"Failed to initialize ML endpoint: {str(e)}")
            print("Running in fallback mode with mock predictions")
            self.is_available = False
    
    def predict_delay(self, train_number: str, source_station: str, destination_station: str) -> dict:
        """
        Predict delay for a train segment
        Args:
            train_number: Train number
            source_station: Source station code (e.g., 'NDLS' from 'NDLS_NewDelhi')
            destination_station: Destination station code (e.g., 'CPR' from 'CPR_Chapra')
        Returns:
            Dictionary with prediction results or fallback prediction if ML endpoint is unavailable
        """
        if not self.is_available:
            return self._get_fallback_prediction()
            
        try:
            # Extract station codes if full station names are provided
            source_code = source_station.split('_')[0] if '_' in source_station else source_station
            dest_code = destination_station.split('_')[0] if '_' in destination_station else destination_station
            
            instance = {
                'Train_Number': str(train_number),
                'Source_Station': source_code,
                'Destination_Station': dest_code
            }
            
            response = self.endpoint.predict(instances=[instance])
            prediction = response.predictions[0]
            
            return {
                'predicted_delay': prediction['value'],
                'min_delay': prediction['lower_bound'],
                'max_delay': prediction['upper_bound'],
                'confidence_level': self._calculate_confidence_level(prediction['value'])
            }
        except Exception as e:
            logger.error(f"Prediction error for train {train_number}: {str(e)}")
            return None
    
    def _calculate_confidence_level(self, predicted_delay: float) -> str:
        """Calculate confidence level based on predicted delay"""
        if predicted_delay <= 15:
            return "HIGH"
        elif predicted_delay <= 30:
            return "MEDIUM"
        else:
            return "LOW"

    def _get_fallback_prediction(self) -> dict:
        """Provide a fallback prediction when ML endpoint is unavailable"""
        return {
            'predicted_delay': 15,
            'min_delay': 5,
            'max_delay': 25,
            'confidence_level': 'MEDIUM'
        }

def enhance_routes_with_predictions(routes: list, predictor: TrainDelayPredictor) -> list:
    """
    Add delay predictions to each route segment
    """
    for route in routes:
        for segment in route['segments']:
            # Only predict for first train in each segment
            if segment == route['segments'][0]:
                prediction = predictor.predict_delay(
                    segment['train_number'],
                    segment['from_station'],
                    segment['to_station']
                )
                
                if prediction:
                    segment['delay_prediction'] = prediction
    
    return routes