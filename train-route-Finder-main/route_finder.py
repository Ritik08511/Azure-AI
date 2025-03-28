#6
from typing import List, Dict, Set, Tuple, Optional
import logging
from datetime import datetime, timedelta
import re
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from train_stops_store import TrainStopsStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the cache at the module level
stops_store = TrainStopsStore()

def parse_train_time(time_str: str, date_str: str) -> Optional[datetime]:
    """
    Parse time strings from both scraper formats with improved date handling.
    
    Args:
        time_str: Time string from either scraper (e.g., "15 Nov, 13:15" or "13:15")
        date_str: Base date in YYYYMMDD format
    
    Returns:
        datetime object or None if parsing fails
    """
    try:
        
        if time_str.strip() == "Start":
            return datetime.strptime(date_str, "%Y%m%d").replace(hour=0, minute=0)
        elif time_str.strip() == "Finish":
            return datetime.strptime(date_str, "%Y%m%d").replace(hour=23, minute=59)
        
        
        base_date = datetime.strptime(date_str, "%Y%m%d")
        
        
        full_date_match = re.match(r"(\d{1,2})\s+([A-Za-z]{3}),\s*(\d{2}):(\d{2})", time_str.strip())
        if full_date_match:
            day, month, hour, minute = full_date_match.groups()
            
            month_num = datetime.strptime(month, "%b").month
            
            year = base_date.year
            try:
                return datetime(year, month_num, int(day), int(hour), int(minute))
            except ValueError:
                logger.error(f"Invalid date components: year={year}, month={month_num}, day={day}, hour={hour}, minute={minute}")
                return None
        
        
        time_match = re.match(r"(\d{2}):(\d{2})", time_str.strip())
        if time_match:
            hour, minute = map(int, time_match.groups())
            return base_date.replace(hour=hour, minute=minute)
        
        
        extended_match = re.match(r"(\d{2}):(\d{2})\s+(\d{1,2})\s+([A-Za-z]{3})\s+(\d{2})", time_str.strip())
        if extended_match:
            hour, minute, day, month, year = extended_match.groups()
            month_num = datetime.strptime(month, "%b").month
            full_year = 2000 + int(year)  
            return datetime(full_year, month_num, int(day), int(hour), int(minute))
        
        raise ValueError(f"Unrecognized time format: {time_str}")
        
    except Exception as e:
        logger.error(f"Error parsing time: {time_str} with base date {date_str} - {str(e)}")
        logger.debug(f"Full error details:", exc_info=True)
        return None

def get_next_day_date(date_str: str) -> str:
    """Convert YYYYMMDD to next day's date in same format"""
    current_date = datetime.strptime(date_str, "%Y%m%d")
    next_date = current_date + timedelta(days=1)
    return next_date.strftime("%Y%m%d")

def parse_train_details(train_data: Dict, date_str: str) -> Dict:
    """
    Extract standardized train details from either scraper format.
    Returns times as datetime objects and availability status.
    """
    try:
        train_number = train_data.get('number', '')
        
        
        departure_time = None
        arrival_time = None
        departure_date = date_str
        arrival_date = date_str
        
    
        if 'departure_time' in train_data and 'arrival_time' in train_data:
            departure_time = parse_train_time(train_data['departure_time'], date_str)
            arrival_time = parse_train_time(train_data['arrival_time'], date_str)
            
            
            if arrival_time and departure_time and arrival_time < departure_time:
                arrival_date = get_next_day_date(date_str)
                arrival_time = parse_train_time(train_data['arrival_time'], arrival_date)
        
        
        elif 'stops' in train_data:
            first_stop = train_data['stops'][0]
            last_stop = train_data['stops'][-1]
            
            departure_time = parse_train_time(first_stop['departure_time'], date_str)
            arrival_time = parse_train_time(last_stop['arrival_time'], date_str)
            
            
            if 'Dropping Point' in last_stop.get('halt_duration', ''):
                arrival_date = get_next_day_date(date_str)
                arrival_time = parse_train_time(last_stop['arrival_time'], arrival_date)
        
        return {
            'train_number': train_number,
            'departure_time': departure_time,
            'arrival_time': arrival_time,
            'departure_date': departure_date,
            'arrival_date': arrival_date,
            'has_seats': has_available_seats(train_data)
        }
        
    except Exception as e:
        logger.error(f"Error parsing train details: {str(e)}")
        return None

def convert_station_format(station_str: str) -> str:
    """Convert station string to CODE_StationName format"""
    try:
        if '(' not in station_str or ')' not in station_str:
            return station_str
            
        station_name, code = station_str.split('(')
        code = code.strip(')')
        station_name = station_name.strip()
        station_name = ''.join(word.capitalize() for word in station_name.split())
        return f"{code}_{station_name}"
    except Exception as e:
        logger.error(f"Error converting station format: {str(e)}")
        return station_str

def has_available_seats(train_data: Dict) -> bool:
    """Check seat availability across all classes"""
    try:
        
        if 'availability' in train_data:
            return 'AVL' in train_data['availability']
            
        
        if 'classes_and_availability' in train_data:
            for class_info in train_data['classes_and_availability']:
                if class_info.get('availability', '').startswith('AVL'):
                    return True
        if 'availability' in train_data:
            return 'RAC' in train_data['availability']
            
    
        if 'classes_and_availability' in train_data:
            for class_info in train_data['classes_and_availability']:
                if class_info.get('availability', '').startswith('RAC'):
                    return True        
        return False
        
    except Exception as e:
        logger.error(f"Error checking seat availability: {str(e)}")
        return False

def is_valid_connection(arrival_time: datetime, departure_time: datetime, min_connection_time: int = 30) -> bool:
    """Check if connection time is valid (>= min_connection_time minutes)"""
    if not arrival_time or not departure_time:
        return False
        
    time_difference = departure_time - arrival_time
    minutes_difference = time_difference.total_seconds() / 60
    return minutes_difference >= min_connection_time

def process_single_route(origin: str, destination: str, date: str, train: Dict, 
                        scrape_routes, scrape_availability, result_queue: Queue, stop_event: threading.Event,
                        found_routes: set, found_routes_lock: threading.Lock):
    """Process a single train route in a separate thread"""
    train_number = train.get('number', 'Unknown')
    thread_name = threading.current_thread().name
    
    try:
        if stop_event.is_set():
            return

        # Get route data
        cached_stops = stops_store.get_stops(train_number)
        if cached_stops:
            logger.info(f"Thread {thread_name}: Using cached stops for train {train_number}")
            route_data = [{'number': train_number, 'stops': cached_stops}]
        else:
            logger.info(f"Thread {thread_name}: Fetching route data for train {train_number}")
            route_data = scrape_routes(origin, destination, date, target_train_number=train_number)
            if route_data and route_data[0].get('stops'):
                stops_store.add_stops(train_number, route_data[0]['stops'])
                
        if not route_data:
            return
            
        processed_stations = set()
        
        for train_route in route_data:
            if stop_event.is_set():
                return
                
            if 'stops' not in train_route:
                continue
                
            logger.info(f"Thread {thread_name}: Processing {len(train_route['stops'])} stops")
            
            for stop in train_route['stops']:
                if stop_event.is_set():
                    return
                    
                intermediate = (
                    convert_station_format(f"{stop['station_name']} ({stop['station_code']})")
                    if 'station_code' in stop
                    else convert_station_format(stop['station_name'])
                )
                
                if intermediate in processed_stations or intermediate in (origin, destination):
                    continue
                    
                processed_stations.add(intermediate)
                
                arrival_time = parse_train_time(stop['arrival_time'], date)
                if not arrival_time:
                    continue
                
                connection_date = date
                if arrival_time.hour >= 23 and arrival_time.minute >= 30:
                    connection_date = get_next_day_date(date)
                
                # Quick check for available seats to destination
                logger.info(f"Thread {thread_name}: Checking seat availability from {intermediate} to {destination}")
                second_leg_trains = scrape_availability(intermediate, destination, connection_date)
                
                available_second_legs = []
                # Check same day
                if second_leg_trains:
                    for train in second_leg_trains:
                        details = parse_train_details(train, connection_date)
                        if details and details['has_seats']:
                            available_second_legs.append((train, connection_date))
                
                # Check next day if needed
                if not available_second_legs:
                    next_day = get_next_day_date(connection_date)
                    second_leg_trains = scrape_availability(intermediate, destination, next_day)
                    if second_leg_trains:
                        for train in second_leg_trains:
                            details = parse_train_details(train, next_day)
                            if details and details['has_seats']:
                                available_second_legs.append((train, next_day))
                
                if not available_second_legs:
                    logger.info(f"Thread {thread_name}: No seats available from {intermediate} to {destination}, skipping station")
                    continue
                
                # If we have available seats, check first leg trains
                logger.info(f"Thread {thread_name}: Found {len(available_second_legs)} trains with seats from {intermediate}")
                first_leg_trains = scrape_availability(origin, intermediate, date)
                
                for first_leg in first_leg_trains:
                    if stop_event.is_set():
                        return
                        
                    first_leg_details = parse_train_details(first_leg, date)
                    if not first_leg_details or not first_leg_details['has_seats']:
                        continue
                    
                    # Check each available second leg
                    for second_leg, second_leg_date in available_second_legs:
                        if stop_event.is_set():
                            return
                            
                        second_leg_details = parse_train_details(second_leg, second_leg_date)
                        
                        # Create unique route identifier
                        route_key = (
                            first_leg_details['train_number'],
                            intermediate,
                            second_leg_details['train_number']
                        )
                        
                        with found_routes_lock:
                            if route_key in found_routes:
                                continue
                                
                            if is_valid_connection(
                                first_leg_details['arrival_time'],
                                second_leg_details['departure_time']
                            ):
                                found_routes.add(route_key)
                                logger.info(f"Thread {thread_name}: Found valid route via {intermediate}")
                                
                                # Extract pure station codes for ML model
                                origin_code = origin.split('_')[0] if '_' in origin else origin
                                dest_code = destination.split('_')[0] if '_' in destination else destination
                                intermediate_code = intermediate.split('_')[0] if '_' in intermediate else intermediate
                                
                                # Update ML model synchronously to ensure it's saved
                                try:
                                    from train_route_learner import FastTrainRoutePredictor
                                    predictor = FastTrainRoutePredictor('train_stops.json', 'station_names.json')
                                    
                                    # Update the route with just the codes
                                    predictor.station_stats[intermediate_code]['success_count'] += 1
                                    predictor.station_stats[intermediate_code]['route_successes'][(origin_code, dest_code)] += 1
                                    
                                    # Save to the JSON file directly (not async)
                                    predictor.save_successful_routes('successful_routes.json')
                                    logger.info(f"Thread {thread_name}: Saved successful route via {intermediate_code}")
                                except Exception as e:
                                    logger.error(f"Thread {thread_name}: Error saving route: {str(e)}")
                                
                                route = {
                                    'segments': [
                                        {
                                            'train_number': first_leg_details['train_number'],
                                            'from_station': origin,
                                            'to_station': intermediate,
                                            'departure_time': first_leg_details['departure_time'],
                                            'arrival_time': first_leg_details['arrival_time'],
                                            'departure_date': first_leg_details['departure_date'],
                                            'arrival_date': first_leg_details['arrival_date']
                                        },
                                        {
                                            'train_number': second_leg_details['train_number'],
                                            'from_station': intermediate,
                                            'to_station': destination,
                                            'departure_time': second_leg_details['departure_time'],
                                            'arrival_time': second_leg_details['arrival_time'],
                                            'departure_date': second_leg_details['departure_date'],
                                            'arrival_date': second_leg_details['arrival_date']
                                        }
                                    ]
                                }
                                result_queue.put(route)
                                return  # Exit after finding a valid route

    except Exception as e:
        logger.error(f"Thread {thread_name}: Error processing train {train_number}: {str(e)}")
        logger.debug(f"Thread {thread_name}: Full error details:", exc_info=True)

def find_routes(origin: str, destination: str, date: str, scrape_availability, scrape_routes, max_routes: int = 1):
    """
    Find routes between stations with valid connections and seat availability.
    Uses parallel processing for faster results
    """
    logger.info(f"Finding up to {max_routes} routes from {origin} to {destination} on {date}")
    
    all_routes = []
    result_queue = Queue()
    stop_event = threading.Event()
    found_routes = set()
    found_routes_lock = threading.Lock()
    executor = None
    futures = []
    
    def quick_shutdown():
        """Immediately shutdown everything"""
        stop_event.set()
        if executor:
            executor._threads.clear()
            for f in futures:
                f.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
        return all_routes[:max_routes]  # Return immediately with found routes
    
    def process_ml_station(intermediate, origin, destination, date, 
                      scrape_routes, scrape_availability, result_queue, stop_event,
                      found_routes, found_routes_lock):
        """Process a single ML-predicted intermediate station using similar logic to process_single_route"""
        thread_name = threading.current_thread().name
        
        try:
            if stop_event.is_set():
                return
                
            logger.info(f"Thread {thread_name}: Checking ML-predicted station: {intermediate}")
            
            # Quick check for available seats from intermediate to destination
            logger.info(f"Thread {thread_name}: Checking seat availability from {intermediate} to {destination}")
            second_leg_trains = scrape_availability(intermediate, destination, date)
            
            available_second_legs = []
            # Check same day
            if second_leg_trains:
                for train in second_leg_trains:
                    if stop_event.is_set():
                        return
                    details = parse_train_details(train, date)
                    if details and details['has_seats']:
                        available_second_legs.append((train, date))
            
            # Check next day if needed
            if not available_second_legs:
                next_day = get_next_day_date(date)
                second_leg_trains = scrape_availability(intermediate, destination, next_day)
                if second_leg_trains:
                    for train in second_leg_trains:
                        if stop_event.is_set():
                            return
                        details = parse_train_details(train, next_day)
                        if details and details['has_seats']:
                            available_second_legs.append((train, next_day))
            
            if not available_second_legs:
                logger.info(f"Thread {thread_name}: No seats available from {intermediate} to {destination}, skipping station")
                return
            
            # If we have available seats for second leg, check first leg trains
            logger.info(f"Thread {thread_name}: Found {len(available_second_legs)} trains with seats from {intermediate}")
            first_leg_trains = scrape_availability(origin, intermediate, date)
            
            for first_leg in first_leg_trains:
                if stop_event.is_set():
                    return
                    
                first_leg_details = parse_train_details(first_leg, date)
                if not first_leg_details or not first_leg_details['has_seats']:
                    continue
                
                # Check each available second leg
                for second_leg, second_leg_date in available_second_legs:
                    if stop_event.is_set():
                        return
                        
                    second_leg_details = parse_train_details(second_leg, second_leg_date)
                    
                    # Create unique route identifier
                    route_key = (
                        first_leg_details['train_number'],
                        intermediate,
                        second_leg_details['train_number']
                    )
                    
                    with found_routes_lock:
                        if route_key in found_routes:
                            continue
                            
                        if is_valid_connection(
                            first_leg_details['arrival_time'],
                            second_leg_details['departure_time']
                        ):
                            found_routes.add(route_key)
                            logger.info(f"Thread {thread_name}: Found valid ML route via {intermediate}")
                            
                            # Extract pure station codes for ML model
                            origin_code = origin.split('_')[0] if '_' in origin else origin
                            dest_code = destination.split('_')[0] if '_' in destination else destination
                            intermediate_code = intermediate.split('_')[0] if '_' in intermediate else intermediate
                            
                            # Update ML model synchronously to ensure it's saved
                            try:
                                from train_route_learner import FastTrainRoutePredictor
                                predictor = FastTrainRoutePredictor('train_stops.json', 'station_names.json')
                                
                                # Update the route with just the codes
                                predictor.station_stats[intermediate_code]['success_count'] += 1
                                predictor.station_stats[intermediate_code]['route_successes'][(origin_code, dest_code)] += 1
                                
                                # Save to the JSON file directly (not async)
                                predictor.save_successful_routes('successful_routes.json')
                                logger.info(f"Thread {thread_name}: Saved successful route via {intermediate_code}")
                            except Exception as e:
                                logger.error(f"Thread {thread_name}: Error saving route: {str(e)}")
                            
                            route = {
                                'segments': [
                                    {
                                        'train_number': first_leg_details['train_number'],
                                        'from_station': origin,
                                        'to_station': intermediate,
                                        'departure_time': first_leg_details['departure_time'],
                                        'arrival_time': first_leg_details['arrival_time'],
                                        'departure_date': first_leg_details['departure_date'],
                                        'arrival_date': first_leg_details['arrival_date']
                                    },
                                    {
                                        'train_number': second_leg_details['train_number'],
                                        'from_station': intermediate,
                                        'to_station': destination,
                                        'departure_time': second_leg_details['departure_time'],
                                        'arrival_time': second_leg_details['arrival_time'],
                                        'departure_date': second_leg_details['departure_date'],
                                        'arrival_date': second_leg_details['arrival_date']
                                    }
                                ]
                            }
                            result_queue.put(route)
                            return  # Exit after finding a valid route

        except Exception as e:
            logger.error(f"Thread {thread_name}: Error processing ML station {intermediate}: {str(e)}")
            logger.debug(f"Thread {thread_name}: Full error details:", exc_info=True)

    
    
    try:
        # Step 1: Check direct trains
        logger.info("Checking direct trains...")
        direct_trains = scrape_availability(origin, destination, date)
        
        if direct_trains:
            logger.info(f"Found {len(direct_trains)} direct trains")
            for train in direct_trains:
                train_details = parse_train_details(train, date)
                if train_details and train_details['has_seats']:
                    logger.info(f"Found direct route with train {train_details['train_number']}")
                    all_routes.append({
                        'segments': [{
                            'train_number': train_details['train_number'],
                            'from_station': origin,
                            'to_station': destination,
                            'departure_time': train_details['departure_time'],
                            'arrival_time': train_details['arrival_time'],
                            'departure_date': train_details['departure_date'],
                            'arrival_date': train_details['arrival_date']
                        }]
                    })
                    if len(all_routes) >= max_routes:
                        logger.info("Found required number of direct routes")
                        return all_routes[:max_routes]  # Early return for direct routes

        # Replace the Step 2 section in find_routes with this improved version
    # Step 2: Try ML-predicted intermediate stations first (with multithreading)
        if len(all_routes) < max_routes:
            logger.info("Trying ML-predicted intermediate stations with multithreading...")
            try:
                # Extract station codes from origin and destination (CODE_StationName format)
                origin_code = origin.split('_')[0] if '_' in origin else origin
                dest_code = destination.split('_')[0] if '_' in destination else destination
                
                # Import the predictor here to avoid circular imports
                from train_route_learner import FastTrainRoutePredictor
                
                # Initialize the predictor
                predictor = FastTrainRoutePredictor('train_stops.json', 'station_names.json')
                
                # Get predicted intermediate stations
                predicted_stations = predictor.predict_intermediate_stations(origin_code, dest_code)
                
                if predicted_stations:
                    logger.info(f"ML predictor suggested {len(predicted_stations)} stations: {predicted_stations}")
                    
                    # Create a thread pool for processing ML stations
                    max_ml_workers = min(5, len(predicted_stations))
                    logger.info(f"Starting {max_ml_workers} worker threads for ML stations")
                    
                    ml_executor = ThreadPoolExecutor(max_workers=max_ml_workers, thread_name_prefix="MLWorker")
                    ml_futures = []
                    
                    # Submit all stations for processing in parallel
                    for station_with_name in predicted_stations:
                        if stop_event.is_set() or len(all_routes) >= max_routes:
                            break
                            
                        # Extract station code from CODE_StationName format
                        intermediate = station_with_name if '_' not in station_with_name else station_with_name
                        
                        # Submit the station for processing with the same parameters as process_single_route
                        future = ml_executor.submit(
                            process_ml_station,
                            intermediate, origin, destination, date,
                            scrape_routes, scrape_availability, result_queue, stop_event,
                            found_routes, found_routes_lock
                        )
                        ml_futures.append(future)
                    
                    # Process results from the result queue immediately for faster response
                    check_start = time.time()
                    while time.time() - check_start < 5:  # Check for up to 5 seconds
                        if stop_event.is_set() or len(all_routes) >= max_routes:
                            break
                            
                        try:
                            route = result_queue.get(timeout=0.1)  # Very short timeout
                            all_routes.append(route)
                            
                            # Extract station code for ML model update
                            intermediate = route['segments'][1]['from_station']
                            intermediate_code = intermediate.split('_')[0] if '_' in intermediate else intermediate
                            
                            # Update the ML model with successful route
                            predictor.update_route_async(origin_code, dest_code, intermediate_code)
                            # Add this line to also save immediately in the main thread:
                            predictor.save_successful_routes('successful_routes.json')
                            
                            logger.info(f"Added ML-predicted route, now have {len(all_routes)}/{max_routes}")
                            
                            if len(all_routes) >= max_routes:
                                logger.info("Found required number of routes using ML predictions")
                                ml_executor.shutdown(wait=False, cancel_futures=True)
                                return quick_shutdown()
                        except Exception:
                            # Check if all futures are done
                            if all(future.done() for future in ml_futures):
                                break
                    
                    # Clean up ML executor
                    ml_executor.shutdown(wait=False, cancel_futures=True)
                    
            except Exception as e:
                logger.error(f"Error using ML predictor: {str(e)}")
                logger.debug("Full error details:", exc_info=True)

        # Step 3: Fall back to original multi-segment route finding if needed
        if len(all_routes) < max_routes:
            logger.info("Checking multi-segment routes using original algorithm...")
            origin_trains = scrape_availability(origin, destination, date)
            if not origin_trains:
                return all_routes
                
            max_workers = min(3, len(origin_trains))
            logger.info(f"Starting {max_workers} worker threads for {len(origin_trains)} trains")
            
            executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="RouteWorker")
            
            for idx, train in enumerate(origin_trains):
                if len(all_routes) >= max_routes:
                    return quick_shutdown()
                    
                future = executor.submit(
                    process_single_route,
                    origin, destination, date, train,
                    scrape_routes, scrape_availability, result_queue, stop_event,
                    found_routes, found_routes_lock
                )
                futures.append(future)
                
                # Check queue immediately after each submission
                while not result_queue.empty():
                    route = result_queue.get_nowait()
                    all_routes.append(route)
                    logger.info(f"Found route {len(all_routes)}/{max_routes}")
                    if len(all_routes) >= max_routes:
                        logger.info("Required number of routes found, shutting down immediately")
                        return quick_shutdown()
            
            # Quick check for remaining results
            while len(all_routes) < max_routes:
                try:
                    route = result_queue.get(timeout=0.1)  # Very short timeout
                    all_routes.append(route)
                    if len(all_routes) >= max_routes:
                        return quick_shutdown()
                except Exception:
                    if all(future.done() for future in futures):
                        break
        
        return all_routes[:max_routes]
        
    finally:
        quick_shutdown()  # Ensure cleanup in all cases

def print_routes(routes: List[Dict]):
    """Print routes with detailed timing information"""
    if not routes:
        print("No routes found with available seats.")
        return
        
    print(f"\nFound {len(routes)} Routes:")
    for i, route in enumerate(routes, 1):
        print(f"\nRoute {i}:")
        for j, segment in enumerate(route['segments'], 1):
            print(f"Segment {j}: Train {segment['train_number']}")
            print(f"From: {segment['from_station']}")
            print(f"To: {segment['to_station']}")
            print(f"Departure: {segment['departure_time'].strftime('%H:%M')} {segment['departure_date']}")
            print(f"Arrival: {segment['arrival_time'].strftime('%H:%M')} {segment['arrival_date']}")
        print("-" * 50)

if __name__ == "__main__":
    origin = "KOAA_Kolkata"
    destination = "DLI_Delhi"
    date = "20250329"
    
    from train_availability_scraper import scrape_train_data as scrape_availability
    from train_route_scraper import scrape_train_routes as scrape_routes
    
    routes = find_routes(origin, destination, date, scrape_availability, scrape_routes, max_routes=1)
    print_routes(routes)
    
    # Print cache statistics
    stats = stops_store.get_cache_stats()
    print("\nCache Statistics:")
    print(f"Total trains in cache: {stats['total_trains']}")
    print(f"Cache hits: {stats['hits']}")
    print(f"Cache misses: {stats['misses']}")