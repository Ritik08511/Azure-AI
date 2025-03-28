import random
import time
import argparse
import gc
import threading
import os
import concurrent.futures
from datetime import datetime, timedelta
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# Try to import psutil, but continue if not available
try:
    import psutil
    has_psutil = True
except ImportError:
    has_psutil = False
    print("Warning: psutil module not installed. Memory monitoring will be limited.")

# Import your modules
from train_availability_scraper import scrape_train_data as scrape_availability
from train_route_scraper import scrape_train_routes as scrape_routes
from route_finder import find_routes, print_routes

# Station data - common Indian railway stations
# Format: CODE_StationName
MAJOR_STATIONS = [
    "NDLS_NewDelhi", "MMCT_MumbaiCentral", "MAS_Chennai", "HWH_Howrah", "SBC_Bengaluru",
    "PUNE_Pune", "ADI_Ahmedabad", "JAT_JammuTawi", "CSTM_MumbaiCST", "LTT_LokmanyaTilak",
    "BZA_Vijayawada", "SC_Secunderabad", "CNB_KanpurCentral", "BPL_Bhopal", "PRYJ_Prayagraj",
    "JP_Jaipur", "BBS_Bhubaneswar", "TVM_Thiruvananthapuram", "RNC_Ranchi", "PNBE_Patna",
    "DLI_Delhi", "GWL_Gwalior", "CDG_Chandigarh", "GZB_Ghaziabad", "R_Raipur",
    "VSG_VascoDaGama", "HYB_Hyderabad", "LKO_Lucknow", "GKP_Gorakhpur", "BSB_Varanasi",
    "AGC_AgraCantt", "DDU_DDUJunction", "BJU_Barauni", "ASR_Amritsar", "JHS_Jhansi"
]

SECONDARY_STATIONS = [
    "KGP_Kharagpur", "MTJ_Mathura", "DBG_Darbhanga", "BE_Bareilly", "SVDK_ShriMataVaishnoDeviKatra",
    "DDN_Dehradun", "RBL_RaeBareli", "BKN_Bikaner", "JU_Jodhpur", "BKSC_BokaroSteel",
    "RPH_Rampur", "LJN_LucknowJunction", "DHN_Dhanbad", "UDZ_Udaipur", "SEE_Sonpur"
]

ALL_STATIONS = MAJOR_STATIONS + SECONDARY_STATIONS

# Function to clean up resources
def cleanup_resources():
    """Force cleanup of resources and threads"""
    # Force garbage collection
    gc.collect()
    
    # Print memory usage
    if has_psutil:
        current_process = psutil.Process(os.getpid())
        print(f"Current memory usage: {current_process.memory_info().rss / 1024 / 1024:.2f} MB")
    
    # Print thread count
    print(f"Active threads: {threading.active_count()}")
    
    # Optional: List all threads
    if threading.active_count() > 5:  # If there are more than expected threads
        print("Active threads:")
        for thread in threading.enumerate():
            print(f" - {thread.name} (daemon: {thread.daemon})")
    
    # Let the OS know we're done with any non-essential memory
    gc.collect()


def generate_date_range(days_ahead: int = 7) -> List[str]:
    """Generate a list of dates for testing"""
    dates = []
    today = datetime.now()
    for i in range(2, days_ahead):  # Start from 7 days ahead (for more likely availability)
        date = today + timedelta(days=i)
        dates.append(date.strftime("%Y%m%d"))
    return dates


def generate_station_pairs(num_pairs: int) -> List[Tuple[str, str, str]]:
    """Generate random station pairs with dates
    
    Returns:
        List of (origin, destination, date) tuples
    """
    pairs = []
    dates = generate_date_range()
    
    # Generate pairs
    while len(pairs) < num_pairs:
        # Pick origin
        origin = random.choice(ALL_STATIONS)
        
        # Pick destination (not the same as origin)
        destinations = [s for s in ALL_STATIONS if s != origin]
        destination = random.choice(destinations)
        
        # Pick a random date
        date = random.choice(dates)
        
        pairs.append((origin, destination, date))
    
    return pairs


# Cross-platform timeout function using ThreadPoolExecutor
def find_routes_with_timeout(origin, destination, date, scrape_availability_func, 
                            scrape_routes_func, max_routes=1, timeout_seconds=120):
    """Run find_routes with a timeout using ThreadPoolExecutor"""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            find_routes, 
            origin, destination, date, 
            scrape_availability_func, scrape_routes_func, 
            max_routes=max_routes
        )
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            # This will cancel the future if possible
            future.cancel()
            raise TimeoutError(f"Route finding timed out after {timeout_seconds} seconds")


def run_station_tests(num_pairs: int, delay: int, max_routes: int = 1, timeout_seconds: int = 120):
    """Run the route finder for different station combinations with resource cleanup"""
    print(f"Testing {num_pairs} random station pairs with {delay} seconds delay between tests")
    print(f"Searching for up to {max_routes} routes per pair")
    print("-" * 50)
    
    pairs = generate_station_pairs(num_pairs)
    
    for i, (origin, destination, date) in enumerate(pairs, 1):
        print(f"\nTest {i}/{num_pairs}:")
        print(f"Finding routes from {origin} to {destination} on {date}")
        
        try:
            # Call find_routes with timeout
            routes = find_routes_with_timeout(
                origin, destination, date,
                scrape_availability, scrape_routes,
                max_routes=max_routes,
                timeout_seconds=timeout_seconds
            )
            
            # Print the results
            print_routes(routes)
            
            # Additional summary
            if routes:
                print(f"Found {len(routes)} routes")
                for route in routes:
                    num_segments = len(route['segments'])
                    if num_segments > 1:
                        intermediate = route['segments'][0]['to_station']
                        print(f"Route with {num_segments} segments via {intermediate}")
            else:
                print("No routes found")
                
        except TimeoutError as e:
            print(f"Timeout error: {str(e)}")
        except Exception as e:
            print(f"Error finding routes: {str(e)}")
        
        print("-" * 50)
        
        # Clean up resources after each test
        print("Cleaning up resources...")
        cleanup_resources()
        
        # Wait before the next request to avoid overwhelming the server
        if i < num_pairs:
            print(f"Waiting {delay} seconds before next test...")
            time.sleep(delay)


def run_specific_pairs(pairs, delay, max_routes=1, timeout_seconds=120):
    """Run tests for specific station pairs with resource cleanup"""
    print(f"Testing {len(pairs)} specific station pairs with {delay} seconds delay between tests")
    
    for i, (origin, destination, date) in enumerate(pairs, 1):
        print(f"\nTest {i}/{len(pairs)}:")
        print(f"Finding routes from {origin} to {destination} on {date}")
        
        try:
            # Call find_routes with timeout
            routes = find_routes_with_timeout(
                origin, destination, date,
                scrape_availability, scrape_routes,
                max_routes=max_routes,
                timeout_seconds=timeout_seconds
            )
            print_routes(routes)
        except TimeoutError as e:
            print(f"Timeout error: {str(e)}")
        except Exception as e:
            print(f"Error finding routes: {str(e)}")
        
        print("-" * 50)
        
        # Clean up resources after each test
        print("Cleaning up resources...")
        cleanup_resources()
        
        if i < len(pairs):
            print(f"Waiting {delay} seconds before next test...")
            time.sleep(delay)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test different station combinations")
    parser.add_argument("--pairs", type=int, default=100, help="Number of random station pairs to test")
    parser.add_argument("--delay", type=int, default=300, help="Delay between tests in seconds")
    parser.add_argument("--routes", type=int, default=1, help="Maximum routes to find per pair")
    parser.add_argument("--mode", choices=["random", "specific"], default="random", 
                      help="Mode: 'random' for random pairs, 'specific' for predefined pairs")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout in seconds for each route finding")
    args = parser.parse_args()
    
    if args.mode == "random":
        run_station_tests(args.pairs, args.delay, args.routes, args.timeout)
    else:
        # Define specific station pairs to test
        specific_pairs = [
            # Long distance major pairs
            ("NDLS_NewDelhi", "MAS_Chennai", "20250320"),
            ("HWH_Howrah", "BCT_Mumbai", "20250325"),
            ("SBC_Bengaluru", "NDLS_NewDelhi", "20250330"),
            
            # Medium distance pairs
            ("JP_Jaipur", "BSB_Varanasi", "20250322"),
            ("PNBE_Patna", "SBC_Bengaluru", "20250326"),
            
            # Short distance pairs
            ("NDLS_NewDelhi", "AGC_Agra", "20250318"),
            ("CNB_Kanpur", "LKO_Lucknow", "20250321"),
            
            # Pairs with secondary stations
            ("NDLS_NewDelhi", "CDG_Chandigarh", "20250324"),
            ("DBG_Darbhanga", "PNBE_Patna", "20250327"),
            ("UHL_Dehradun", "NDLS_NewDelhi", "20250329")
        ]
        run_specific_pairs(specific_pairs, args.delay, args.routes, args.timeout)
    
    # Final cleanup
    print("\nFinal resource cleanup...")
    cleanup_resources()
    print("All tests completed.")