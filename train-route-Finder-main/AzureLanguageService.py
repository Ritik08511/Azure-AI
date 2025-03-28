import requests
import json
import time
from datetime import datetime, timedelta
import re

# Replace with your values
prediction_url = "https://nlptrain.cognitiveservices.azure.com/language/analyze-text/jobs?api-version=2024-11-15-preview"
api_key = "70366kZzATdU6OnYaDOerEKvZBpYIWGr9KWAOJbLsxdQwnnTwSi2JQQJ99BCACYeBjFXJ3w3AAAaACOGmzh8"  # You should have this from your Azure portal
project_name = "trainner"
deployment_name = "train-booking-deployment"  # Replace with your actual deployment name  # The name you gave your deployment

# Add this function to your code
def get_station_code_and_name(station_text):
    """
    Convert station name to code_name format or extract code and name
    from code_name format.
    
    Args:
        station_text: Text identified as Origin or Destination
        
    Returns:
        tuple: (station_code, station_name)
    """
    # Dictionary mapping station names to codes
    # Replace this with your full list of stations
    station_mapping = {
        "Delhi": "DLI",
        "Mumbai": "CSTM",
        "Chennai": "MAS",
        "Kolkata": "KOAA",
        "Bangalore": "SBC",
        "Hyderabad": "SC",
        "Pune": "PUNE",
        "Jaipur": "JP",
        "Ahmedabad": "ADI",
        "Lucknow": "LKO",
        # Add more stations as needed
    }
    
    # Check if input is already in code_name format
    if "_" in station_text:
        parts = station_text.split("_", 1)
        return parts[0], parts[1]
    
    # Try to find the station in our mapping
    for name, code in station_mapping.items():
        if name.lower() == station_text.lower():
            return code, name
    
    # If we can't find a match, return the original text
    return None, station_text

def extract_booking_details(query):
    # Prepare the request headers
    headers = {
        "Content-Type": "application/json",
        "Ocp-Apim-Subscription-Key": api_key
    }
    
    # Prepare the request body - fixing the format to include tasks
    body = {
        "analysisInput": {
            "documents": [
                {
                    "id": "1",
                    "text": query
                }
            ]
        },
        "tasks": [
            {
                "kind": "CustomEntityRecognition",
                "parameters": {
                    "projectName": project_name,
                    "deploymentName": deployment_name
                }
            }
        ]
    }
    
    # Submit the analysis job
    response = requests.post(prediction_url, headers=headers, json=body)
    
    if response.status_code == 202:  # Accepted
        # Get the operation location for polling
        operation_location = response.headers["Operation-Location"]
        
        # Poll until the job is complete
        while True:
            poll_response = requests.get(operation_location, headers={"Ocp-Apim-Subscription-Key": api_key})
            poll_result = poll_response.json()
            
            if poll_result["status"] == "succeeded":
                # Extract the results
                # Check the structure of poll_result to find where the entities are
                print("Response structure:", json.dumps(poll_result, indent=2))
                
                # Assuming the correct structure based on typical Azure responses
                results = poll_result.get("tasks", {}).get("items", [{}])[0].get("results", {})
                
                break
            elif poll_result["status"] == "failed":
                raise Exception(f"Analysis failed: {poll_result.get('error', {}).get('message', 'Unknown error')}")
            
            # Wait before polling again
            time.sleep(1)
        
        # Extract entities
        booking_details = {"origin": None, "destination": None, "journey_date": None}
        
        # Adjust this part based on the printed structure
        for document in results.get("documents", []):
            for entity in document.get("entities", []):
                if entity["category"] == "Origin":
                    origin_text = entity["text"]
                    origin_code, origin_name = get_station_code_and_name(origin_text)
                    booking_details["origin"] = f"{origin_code}_{origin_name}" if origin_code else origin_text
                elif entity["category"] == "Destination":
                    dest_text = entity["text"]
                    dest_code, dest_name = get_station_code_and_name(dest_text)
                    booking_details["destination"] = f"{dest_code}_{dest_name}" if dest_code else dest_text
                elif entity["category"] == "JourneyDate":
                    booking_details["journey_date"] = entity["text"]
        
        # Convert journey_date to YYYYMMDD format if needed
        if booking_details["journey_date"]:
            booking_details["formatted_date"] = parse_date_expression(booking_details["journey_date"])
        
        return booking_details
    else:
        raise Exception(f"Request failed with status code {response.status_code}: {response.text}")

def parse_date_expression(date_text):
    """
    Convert various date formats to YYYYMMDD format
    """
    today = datetime.now()
    date_text = date_text.lower().strip()
    
    # Handle "today" and "tomorrow"
    if "today" in date_text:
        return today.strftime("%Y%m%d")
    elif "tomorrow" in date_text:
        return (today + timedelta(days=1)).strftime("%Y%m%d")
    
    # Handle "this [day]" and "next [day]"
    days = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, 
            "friday": 4, "saturday": 5, "sunday": 6}
    
    for day, day_num in days.items():
        if day in date_text:
            current_day_num = today.weekday()
            days_ahead = 0
            
            if "next" in date_text:
                # "Next [day]" means the [day] in the following week
                days_ahead = (day_num - current_day_num) % 7
                if days_ahead == 0:
                    days_ahead = 7  # Next same day means a week later
            else:
                # "This [day]" or just "[day]" means the coming day
                days_ahead = (day_num - current_day_num) % 7
                if days_ahead == 0 and "this" not in date_text:
                    days_ahead = 7  # Next week if just the day name
            
            target_date = today + timedelta(days=days_ahead)
            return target_date.strftime("%Y%m%d")
    
    # Try to parse formatted dates
    try:
        # Try common date formats
        for fmt in ["%d %B", "%d %b", "%B %d", "%b %d", "%d/%m", "%m/%d"]:
            try:
                # Add current year if not specified
                parsed_date = datetime.strptime(date_text, fmt).replace(year=today.year)
                # If the date is in the past, it might refer to next year
                if parsed_date < today and (today - parsed_date).days > 7:
                    parsed_date = parsed_date.replace(year=today.year + 1)
                return parsed_date.strftime("%Y%m%d")
            except ValueError:
                continue
        
        # Try formats with year
        for fmt in ["%d %B %Y", "%B %d %Y", "%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d"]:
            try:
                parsed_date = datetime.strptime(date_text, fmt)
                return parsed_date.strftime("%Y%m%d")
            except ValueError:
                continue
                
        # Try to extract with regex
        date_match = re.search(r"(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]+)", date_text)
        if date_match:
            day, month = date_match.groups()
            month_names = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
            for i, m in enumerate(month_names):
                if m in month.lower()[:3]:
                    month_num = i + 1
                    parsed_date = datetime(today.year, month_num, int(day))
                    if parsed_date < today:
                        parsed_date = parsed_date.replace(year=today.year + 1)
                    return parsed_date.strftime("%Y%m%d")
    
    except Exception as e:
        print(f"Error parsing date '{date_text}': {str(e)}")
    
    # If all parsing attempts fail, return None
    return None

# Example usage
query = "I want to book a train from Delhi to Mumbai next Sunday"
result = extract_booking_details(query)
print(f"Origin: {result['origin']}")
print(f"Destination: {result['destination']}")
print(f"Journey Date (original): {result['journey_date']}")
print(f"Journey Date (formatted): {result['formatted_date']}")