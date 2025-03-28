from flask import Flask, render_template, request, jsonify
import logging
from datetime import datetime, timedelta
from route_finder import find_routes, print_routes  
from train_availability_scraper import scrape_train_data
from train_route_scraper import scrape_train_routes
from pyngrok import ngrok, conf
import os
import atexit
import signal
import sys
import requests
import json
import time
import re
from delay_prediction_module import TrainDelayPredictor, enhance_routes_with_predictions
from flask_cors import CORS

# Constants
NGROK_AUTH_TOKEN = "2s8AbCxwpLLc5wO0kEKsl1VaXGA_57VcrKmDYQuHdpheaigPJ"
AZURE_SPEECH_KEY = "4KKeuEtS2afH6xalV7mHMIrYngmODCXfW6Vgjiv57lsz2kCXzhabJQQJ99BCACYeBjFXJ3w3AAAYACOGSdmp"
AZURE_SPEECH_REGION = "eastus"

# Azure Language Service configuration
AZURE_LANGUAGE_KEY = "70366kZzATdU6OnYaDOerEKvZBpYIWGr9KWAOJbLsxdQwnnTwSi2JQQJ99BCACYeBjFXJ3w3AAAaACOGmzh8"
AZURE_LANGUAGE_ENDPOINT = "https://nlptrain.cognitiveservices.azure.com/language/analyze-text/jobs?api-version=2024-11-15-preview"
AZURE_LANGUAGE_PROJECT = "trainner"
AZURE_LANGUAGE_DEPLOYMENT = "train-booking-deployment"

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "allow_headers": ["Content-Type", "Authorization"],
        "methods": ["GET", "POST", "OPTIONS"]
    }
})

# Global variable to store ngrok tunnel
tunnel = None

def cleanup_ngrok():
    """Cleanup function to kill ngrok process"""
    global tunnel
    if tunnel:
        try:
            ngrok.disconnect(tunnel.public_url)
        except:
            pass
    ngrok.kill()

def init_ngrok():
    """Initialize and start ngrok tunnel"""
    global tunnel
    
    try:
        # Clean up any existing ngrok processes
        cleanup_ngrok()
        
        # Configure ngrok
        conf.get_default().auth_token = NGROK_AUTH_TOKEN
        conf.get_default().region = 'us'
        
        # Set up the tunnel
        tunnel = ngrok.connect(
            addr=5000,
            proto='http',
            bind_tls=True
        )
        
        print(f"\n * Ngrok tunnel established successfully")
        print(f" * Public URL: {tunnel.public_url}")
        return tunnel
        
    except Exception as e:
        print(f"Error setting up ngrok: {str(e)}")
        return None

def signal_handler(sig, frame):
    """Handle cleanup on system signals"""
    cleanup_ngrok()
    sys.exit(0)

# Set up the credentials path - UPDATED to match your working app
CREDENTIALS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "keys",
    "app-engine-key.json"  
)

# Set the environment variable for Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_PATH

# Verify credentials file exists
if not os.path.exists(CREDENTIALS_PATH):
    print(f"Warning: Credentials file not found at: {CREDENTIALS_PATH}")
    # Create directory if it doesn't exist
    try:
        os.makedirs(os.path.dirname(CREDENTIALS_PATH), exist_ok=True)
        print(f"Created directory: {os.path.dirname(CREDENTIALS_PATH)}")
    except Exception as e:
        print(f"Error creating directory: {str(e)}")
else:
    print(f"Using credentials file: {CREDENTIALS_PATH}")

# NER Functions
def extract_booking_details(query):
    """
    Extract booking details from a natural language query using Azure Language Service
    """
    # Prepare the request headers
    headers = {
        "Content-Type": "application/json",
        "Ocp-Apim-Subscription-Key": AZURE_LANGUAGE_KEY
    }
    
    # Prepare the request body
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
                    "projectName": AZURE_LANGUAGE_PROJECT,
                    "deploymentName": AZURE_LANGUAGE_DEPLOYMENT
                }
            }
        ]
    }
    
    # Submit the analysis job
    response = requests.post(AZURE_LANGUAGE_ENDPOINT, headers=headers, json=body)
    
    if response.status_code == 202:  # Accepted
        # Get the operation location for polling
        operation_location = response.headers["Operation-Location"]
        
        # Poll until the job is complete
        while True:
            poll_response = requests.get(operation_location, headers={"Ocp-Apim-Subscription-Key": AZURE_LANGUAGE_KEY})
            poll_result = poll_response.json()
            
            if poll_result["status"] == "succeeded":
                # Extract the results from the correct path in the response
                results = poll_result["tasks"]["items"][0]["results"]
                break
            elif poll_result["status"] == "failed":
                raise Exception(f"Analysis failed: {poll_result.get('error', {}).get('message', 'Unknown error')}")
            
            # Wait before polling again
            time.sleep(1)
        
        # Extract entities
        booking_details = {
            "origin": None, 
            "destination": None, 
            "journey_date": None,
            "journey_date_text": None,
            "formatted_date": None
        }
        
        for document in results["documents"]:
            for entity in document["entities"]:
                if entity["category"] == "Origin":
                    origin_text = entity["text"]
                    formatted_origin = get_station_code_and_name(origin_text)
                    booking_details["origin"] = formatted_origin
                elif entity["category"] == "Destination":
                    dest_text = entity["text"]
                    formatted_dest = get_station_code_and_name(dest_text)
                    booking_details["destination"] = formatted_dest
                elif entity["category"] == "JourneyDate":
                    booking_details["journey_date_text"] = entity["text"]
                    booking_details["journey_date"] = parse_date_expression(entity["text"])
                    if booking_details["journey_date"]:
                        # Format the date for display (YYYY-MM-DD)
                        date_obj = datetime.strptime(booking_details["journey_date"], "%Y%m%d")
                        booking_details["formatted_date"] = date_obj.strftime("%Y-%m-%d")
        
        return booking_details
    else:
        raise Exception(f"Request failed with status code {response.status_code}: {response.text}")

def get_station_code_and_name(station_text):
    """
    Convert a station name to the format CODE_Name
    
    Args:
        station_text (str): The station name or partial name
        
    Returns:
        str: Formatted station string in the format "CODE_Name" or original text if not found
    """
    # Define your station mapping
    # This should be replaced with your actual station data
    station_mapping = {
    "delhi": "DLI_Delhi",
    "new delhi": "NDLS_NewDelhi",  # Updated from NDLS_New Delhi to NDLS_NewDelhi
    "mumbai": "CSTM_Mumbai",
    "mumbai central": "BCT_MumbaiCentral",  # Updated from BCT_Mumbai Central to BCT_MumbaiCentral
    "chennai": "MAS_Chennai",
    "bangalore": "SBC_Bangalore",
    "kolkata": "KOAA_Kolkata",
    "hyderabad": "SC_Hyderabad",
    "pune": "PUNE_Pune",
    "ahmedabad": "ADI_Ahmedabad",
    # Added stations from the list
    "abu road": "ABR_AbuRoad",
    "adilabad": "ADD_Adilabad",
    "adra": "ADF_Adra",
    "adra": "ADRA_Adra",
    "agartala": "AGTE_Agartala",
    "agra cantt": "AGC_AgraCantt",
    "ahmadnagar": "AGN_Ahmadnagar",
    "ajmer": "AII_Ajmer",
    "akola": "AKT_Akola",
    "aligarhi": "ALN_Aligarhi",
    "alipurduar jn": "ALJN_AlipurduarJn",
    "allahabad": "ALD_Allahabad",
    "alappuzha": "ALLP_Alappuzha",
    "aluva": "AWY_Aluva",
    "aurangabad": "AWB_Aurangabad",
    "amalner": "AN_Amalner",
    "amb andaura": "ASAN_AmbAndaura",
    "amethi": "AME_Amethi",
    "ambikapur": "ABKP_Ambikapur",
    "amla": "AMLA_Amla",
    "amritsar": "ASR_Amritsar",
    "anand": "ANA_Anand",
    "anand nagar": "ANDN_AnandNagar",
    "anand vihar terminus": "ANVT_AnandViharTerminus",
    "anantapur": "ATP_Anantapur",
    "andal": "AWL_Andal",
    "ara": "ARA_Ara",
    "anuppur": "APR_Anuppur",
    "arakkonam": "AJJ_Arakkonam",
    "arsikere": "ASK_Arsikere",
    "asansol": "ASN_Asansol",
    "aunrihar": "ARJ_Aunrihar",
    "ayodhya": "AYM_Ayodhya",
    "azamgarh": "AZR_Azamgarh",
    "badarpur": "BPR_Badarpur",
    "badnera": "BDJ_Badnera",
    "belgaum": "BGM_Belgaum",
    "bagipat road": "BPM_BagipatRoad",
    "baidyanathdham": "BDME_Baidyanathdham",
    "bakhtiyarpur": "BKP_Bakhtiyarpur",
    "balasore": "BSPR_Balasore",
    "bangarapet": "BWT_Bangarapet",
    "balangir": "BLGR_Balangir",
    "balaghat": "BALV_Balaghat",
    "balurghat": "BLGT_Balurghat",
    "balipatna": "BPO_Balipatna",
    "bangalore cantt": "BNC_BangaloreCantt",
    "bangalore city": "SBC_BangaloreCity",
    "bankura": "BNK_Bankura",
    "banahat": "BNO_Banahat",
    "bokaro steel city": "BKSC_BokaroSteelCity",
    "bapudham motihari": "BPH_BapudhamMotihari",
    "bandikui": "BAR_Bandikui",
    "baran": "BNJ_Baran",
    "bareilly": "BE_Bareilly",
    "basti": "BST_Basti",
    "bhatinda jn": "BTI_BhatindaJn",
    "bayana": "BYN_Bayana",
    "begu sarai": "BEG_BeguSarai",
    "belapur": "BAP_Belapur",
    "bellary": "BAY_Bellary",
    "bettiah": "BTH_Bettiah",
    "betul": "BEU_Betul",
    "bhopal": "BPL_Bhopal",
    "bhubaneswar": "BBS_Bhubaneswar",
    "bhuj": "BHJ_Bhuj",
    "bhusaval": "BSL_Bhusaval",
    "bijwasan": "BJU_Bijwasan",
    "bikaner": "BKN_Bikaner",
    "borivali": "BVI_Borivali",
    "varanasi": "BSB_Varanasi",
    "coimbatore jn": "CBE_CoimbatoreJn",
    "chandigarh": "CDG_Chandigarh",
    "kanpur central": "CNB_KanpurCentral",
    "chalisgarh": "CSN_Chalisgarh",
    "chennai central": "MAS_ChennaiCentral",
    "chennai egmore": "MS_ChennaiEgmore",
    "dehradun": "DDN_Dehradun",
    "delhi sarai rohilla": "DEE_DelhiSaraiRohilla",
    "delhi shahdara": "DSA_DelhiShahdara",
    "dhanbad": "DHN_Dhanbad",
    "dharmanagar": "DMR_Dharmanagar",
    "dharwad": "DWR_Dharwad",
    "dhone": "DHO_Dhone",
    "darbhanga": "DBG_Darbhanga",
    "darbhanga": "DJ_Darbhanga",
    "durg": "DURG_Durg",
    "erode": "ED_Erode",
    "ernakulam jn": "ERS_ErnakulamJn",
    "etawah": "ETW_Etawah",
    "faizabad": "FD_Faizabad",
    "faridabad": "FBD_Faridabad",
    "fatehpur": "FTP_Fatehpur",
    "firozpur": "FZR_Firozpur",
    "gooty": "G_Gooty",
    "gajraula": "GJL_Gajraula",
    "gorakhpur": "GKP_Gorakhpur",
    "ghaziabad": "GZB_Ghaziabad",
    "guwahati": "GHY_Guwahati",
    "gwalior": "GWL_Gwalior",
    "habibganj": "HBJ_Habibganj",
    "howrah": "HWH_Howrah",
    "indore": "IDH_Indore",
    "ipurupalem": "IPL_Ipurupalem",
    "itwari": "ITR_Itwari",
    "jabalpur": "JBP_Jabalpur",
    "jaipur": "JP_Jaipur",
    "jaisalmer": "JSM_Jaisalmer",
    "jalandhar city": "JUC_JalandharCity",
    "jamnagar": "JAM_Jamnagar",
    "jhansi": "JHS_Jhansi",
    "jodhpur": "JU_Jodhpur",
    "kharagpur": "KGP_Kharagpur",
    "kacheguda": "KCG_Kacheguda",
    "kazipet": "KZJ_Kazipet",
    "kesinga": "KSNG_Kesinga",
    "kendujhargarh": "KDJR_Kendujhargarh",
    "kangasbagh": "KGBS_Kangasbagh",
    "khalilabad": "KLD_Khalilabad",
    "khammam": "KMK_Khammam",
    "khandwa": "KNW_Khandwa",
    "khed": "KEI_Khed",
    "khurda road": "KUR_KhurdaRoad",
    "katihar": "KIR_Katihar",
    "kishanganj": "KNE_Kishanganj",
    "kishangarh": "KSG_Kishangarh",
    "kiul": "KIUL_Kiul",
    "kochuveli": "KCVL_Kochuveli",
    "kodaikanal road": "KQR_KodaikanalRoad",
    "kozhikode": "CLT_Kozhikode",
    "katpadi": "KPD_Katpadi",
    "koraput": "KRPU_Koraput",
    "korba": "KRBA_Korba",
    "kota": "KOTA_Kota",
    "kotdwara": "KTW_Kotdwara",
    "kotkapura": "KCP_Kotkapura",
    "kottayam": "KTYM_Kottayam",
    "krishanagar city": "KNJ_KrishanagarCity",
    "krishnarajapuram": "KJM_Krishnarajapuram",
    "kudal": "KUDL_Kudal",
    "kumbakonam": "KMU_Kumbakonam",
    "kundara": "KUDA_Kundara",
    "kurduvadi": "KWV_Kurduvadi",
    "kurukshetra": "KRNT_Kurukshetra",
    "lakhimpur": "LHU_Lakhimpur",
    "lalkuan": "LL_Lalkuan",
    "lokmanya tilak": "LTT_LokmanyaTilak",
    "lanka": "LKA_Lanka",
    "lucknow": "LKO_Lucknow",
    "lumding": "LMG_Lumding",
    "madgaon": "MAO_Madgaon",
    "madarihat": "MDT_Madarihat",
    "maddur": "MAD_Maddur",
    "makhdumpur": "MDR_Makhdumpur",
    "majhagawan": "MJN_Majhagawan",
    "maliya": "MEB_Maliya",
    "malda town": "MLDT_MaldaTown",
    "manamadurai": "MNM_Manamadurai",
    "mangalore central": "MAQ_MangaloreCentral",
    "mansi": "MNI_Mansi",
    "manmad": "MMR_Manmad",
    "mandwabazar": "MDB_Mandwabazar",
    "meerut city": "MTC_MeerutCity",
    "mettupalayam": "MTP_Mettupalayam",
    "miraj": "MRJ_Miraj",
    "moga": "MOF_Moga",
    "mokama": "MKA_Mokama",
    "mumbai": "MUR_Mumbai",
    "mughalsarai": "MGS_Mughalsarai",
    "muzaffarpur": "MZS_Muzaffarpur",
    "mysore": "MYS_Mysore",
    "nagpur": "NGP_Nagpur",
    "nasik": "NK_Nasik",
    "nagaon": "NGO_Nagaon",
    "narwana": "NDT_Narwana",
    "new coochbehar": "NWP_NewCoochbehar",
    "new farakka": "NFK_NewFarakka",
    "new jalpaiguri": "NJP_NewJalpaiguri",
    "new tinsukia": "NTSK_NewTinsukia",
    "nizamuddin": "NZM_Nizamuddin",
    "ongole": "OGL_Ongole",
    "pachora": "PC_Pachora",
    "palakkad": "PKD_Palakkad",
    "palghat": "PLG_Palghat",
    "panipat": "PNP_Panipat",
    "pathankot": "PTK_Pathankot",
    "patna": "PNBE_Patna",
    "porbandar": "PBR_Porbandar",
    "puri": "PURI_Puri",
    "raipur": "R_Raipur",
    "rameswaram": "RMM_Rameswaram",
    "ranchi": "RNC_Ranchi",
    "ratlam": "RTM_Ratlam",
    "raxaul": "RXL_Raxaul",
    "rewa": "RE_Rewa",
    "rohtak": "ROK_Rohtak",
    "rajendra pul": "RJPB_RajendraPul",
    "sealdah": "SDAH_Sealdah",
    "shimla": "SLI_Shimla",
    "silchar": "SCL_Silchar",
    "solapur": "SLO_Solapur",
    "surat": "ST_Surat",
    "surendra nagar": "SUNR_SurendraNagar",
    "tambaram": "TBM_Tambaram",
    "tiruchchirappalli": "TPJ_Tiruchchirappalli",
    "thiruvananthapuram": "TVC_Thiruvananthapuram",
    "thane": "TNA_Thane",
    "tirupati": "TPTY_Tirupati",
    "udaipur city": "UDZ_UdaipurCity",
    "ujjain": "UJN_Ujjain",
    "ambala": "UMB_Ambala",
    "vijayawada": "BZA_Vijayawada",
    "visakhapatnam": "VSKP_Visakhapatnam",
    "warangal": "WL_Warangal",
    "yesvantpur": "YPR_Yesvantpur",
}
    
    # Normalize the input text
    normalized_text = station_text.lower().strip()
    
    # Try direct match
    if normalized_text in station_mapping:
        return station_mapping[normalized_text]
    
    # Try partial match
    for key, value in station_mapping.items():
        if normalized_text in key or key in normalized_text:
            return value
    
    # No match found
    return station_text

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

@app.route('/', methods=['GET', 'POST'])
def index():
    print(f"Request received: {request.method}")
    
    # For GET requests, just render the form
    if request.method == 'GET':
        print("Rendering index template for GET request")
        return render_template('index.html')
    
    # For POST requests, process the form
    elif request.method == 'POST':
        print("POST request received")
        print(f"Form data: {request.form}")
        
        # Check if we have the required form fields
        if not request.form:
            print("No form data received")
            return render_template('error.html', error="No form data received")
        
        if 'origin' not in request.form or 'destination' not in request.form:
            print("Missing required form fields")
            return render_template('error.html', error="Missing required form fields")
        
        # Extract form data
        origin = request.form['origin']
        destination = request.form['destination']
        date = request.form.get('date', datetime.now().strftime('%Y-%m-%d'))
        max_routes = int(request.form.get('max_routes', 5))
        min_connection_time = int(request.form.get('connection_time', 30))
        
        print(f"Processing route from {origin} to {destination} on {date}")
        
        # Format date
        date = date.replace('-', '')
        
        try:
            # Find routes
            routes = find_routes(
                origin=origin,
                destination=destination,
                date=date,
                scrape_availability=scrape_train_data,
                scrape_routes=scrape_train_routes,
                max_routes=max_routes
            )
            
            print(f"Found {len(routes)} routes")
            
            # Filter routes by connection time
            filtered_routes = []
            for route in routes:
                valid_route = True
                if len(route['segments']) > 1:
                    for i in range(len(route['segments']) - 1):
                        current_segment = route['segments'][i]
                        next_segment = route['segments'][i + 1]
                        time_diff = (next_segment['departure_time'] - current_segment['arrival_time']).total_seconds() / 60
                        if time_diff < min_connection_time:
                            valid_route = False
                            break
                if valid_route:
                    filtered_routes.append(route)
            
            print(f"After filtering, {len(filtered_routes)} routes remain")
            
            # Try to enhance routes with predictions
            error_message = None
            try:
                # Verify credentials before initializing predictor
                print(f"Using credentials from: {CREDENTIALS_PATH}")
                print(f"Credentials file exists: {os.path.exists(CREDENTIALS_PATH)}")
                
                if not os.path.exists(CREDENTIALS_PATH):
                    raise Exception(f"Credentials file not found at: {CREDENTIALS_PATH}")
                
                # Initialize predictor with better error handling
                try:
                    predictor = TrainDelayPredictor(
                        project_id="521902680111",
                        endpoint_id="8318633396381155328",
                        location="us-central1"
                    )
                    print("TrainDelayPredictor initialized successfully")
                except Exception as init_error:
                    raise Exception(f"Failed to initialize predictor: {str(init_error)}")
                
                # Enhance routes with predictions
                print("Adding delay predictions to routes")
                try:
                    enhanced_routes = enhance_routes_with_predictions(filtered_routes, predictor)
                    # Verify predictions were actually added
                    has_predictions = False
                    if enhanced_routes and len(enhanced_routes) > 0:
                        for route in enhanced_routes:
                            for segment in route.get('segments', []):
                                if 'predicted_delay' in segment:
                                    has_predictions = True
                                    break
                            if has_predictions:
                                break
                    
                    if has_predictions:
                        filtered_routes = enhanced_routes
                        print("Successfully enhanced routes with predictions")
                    else:
                        raise Exception("No predictions were added to routes")
                except Exception as pred_error:
                    raise Exception(f"Error enhancing routes: {str(pred_error)}")
                
            except Exception as e:
                error_message = f"Delay prediction error: {str(e)}"
                print(f"Error in prediction: {error_message}")
                # Continue with unenhanced routes
            
            print("Rendering results template")
            return render_template(
                'results.html',
                routes=filtered_routes,
                connection_time=min_connection_time,
                error=error_message
            )
        except Exception as e:
            print(f"Route processing error: {str(e)}")
            import traceback
            traceback.print_exc()
            return render_template('error.html', error=str(e))
    
    # This should never happen, but just in case
    return render_template('error.html', error="Invalid request method")

# New endpoint to get Azure Speech token
@app.route('/api/get-speech-token', methods=['GET', 'OPTIONS'])
def get_speech_token():
    if request.method == 'OPTIONS':
        # Handle preflight request
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,OPTIONS')
        return response

    try:
        headers = {
            'Ocp-Apim-Subscription-Key': AZURE_SPEECH_KEY,
            'Content-Type': 'application/json'
        }
        
        response = requests.post(
            f'https://{AZURE_SPEECH_REGION}.api.cognitive.microsoft.com/sts/v1.0/issueToken',
            headers=headers
        )
        
        if response.status_code == 200:
            resp = jsonify({
                'token': response.text,
                'region': AZURE_SPEECH_REGION
            })
            resp.headers.add('Access-Control-Allow-Origin', '*')
            resp.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
            return resp
        else:
            return jsonify({
                'error': f'Error from Azure: {response.status_code}',
                'message': response.text
            }), 500
            
    except Exception as e:
        print(f"Error getting speech token: {str(e)}")
        return jsonify({
            'error': 'Could not retrieve token',
            'message': str(e)
        }), 500

# Add this route to proxy the Speech SDK script
@app.route('/speech-sdk-proxy', methods=['GET'])
def speech_sdk_proxy():
    try:
        response = requests.get('https://aka.ms/csspeech/jsbrowserpackageraw')
        if response.status_code == 200:
            proxy_response = app.response_class(
                response=response.content,
                status=200,
                mimetype='application/javascript'
            )
            proxy_response.headers.add('Access-Control-Allow-Origin', '*')
            return proxy_response
        else:
            return jsonify({
                'error': f'Error fetching Speech SDK: {response.status_code}'
            }), response.status_code
    except Exception as e:
        return jsonify({
            'error': f'Error proxying Speech SDK: {str(e)}'
        }), 500

# New endpoint for processing natural language queries
@app.route('/api/process-query', methods=['POST', 'OPTIONS'])
def process_query():
    if request.method == 'OPTIONS':
        # Handle preflight request
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'POST,OPTIONS')
        return response
        
    try:
        data = request.json
        query = data.get('query')
        
        if not query:
            return jsonify({"error": "No query provided"}), 400
        
        # Extract booking details using NER
        booking_details = extract_booking_details(query)
        
        return jsonify(booking_details)
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        return jsonify({"error": str(e)}), 500

def run_app():
    """Function to run the Flask app with proper cleanup"""
    # Register cleanup functions
    atexit.register(cleanup_ngrok)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Initialize ngrok
    if init_ngrok():
        # Run the Flask app with reloader disabled
        app.run(debug=True, use_reloader=False)
    else:
        print("Failed to initialize ngrok. Exiting...")
        sys.exit(1)

if __name__ == '__main__':
    run_app()