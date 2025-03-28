from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException
import time
import logging
import random
from train_stops_store import TrainStopsStore
from queue import Queue
from threading import Lock, local
from typing import List, Dict, Optional
import atexit
import threading

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize stops store
stops_store = TrainStopsStore()
driver_pool = Queue(maxsize=5)
driver_lock = Lock()

# Thread-local storage for browser instances
thread_local = local()
browser_instances: Dict[int, webdriver.Chrome] = {}
browser_lock = Lock()

def get_random_user_agent():
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
    ]
    return random.choice(user_agents)

def wait_for_element(driver, by, value, timeout=10):
    return WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located((by, value))
    )

def extract_train_routes(driver, target_train_number=None):
    """
    Extract train routes, optionally filtering for a specific train number.
    Args:
        driver: Selenium WebDriver instance
        target_train_number: Optional train number to filter for
    """
    train_routes = []
    train_elements = driver.find_elements(By.CLASS_NAME, "Gwgxn")
    logger.info(f"Found {len(train_elements)} train elements" + 
                (f", filtering for train {target_train_number}" if target_train_number else ""))

    for train_element in train_elements:
        try:
            train_info = {}
            name_number_element = train_element.find_element(By.CLASS_NAME, "k9j0o")
            train_info['name'] = name_number_element.find_element(By.TAG_NAME, "h1").text
            train_info['number'] = name_number_element.find_element(By.CLASS_NAME, "qW4yv").text.strip("()")
            
            # Skip if not the target train
            if target_train_number and train_info['number'] != target_train_number:
                continue
                
            logger.info(f"Extracting route for train: {train_info['name']} ({train_info['number']})")

            # Find and click the "View train route" button for this specific train
            view_route_button = WebDriverWait(driver, 20).until(
                EC.element_to_be_clickable((By.XPATH, 
                    f"//div[contains(@class, 'Gwgxn') and .//h1[contains(text(), '{train_info['name']}')]]//div[contains(@class, 'iXty4')]"))
            )
            driver.execute_script("arguments[0].scrollIntoView(true);", view_route_button)
            time.sleep(2)
            driver.execute_script("arguments[0].click();", view_route_button)

            # Wait for the route information to load
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'QMO26')]"))
            )

            # Extract route information
            stops = []
            stop_rows = driver.find_elements(By.XPATH, "//div[contains(@class, 'aMT0H')]")
            for row in stop_rows:
                try:
                    stop_info = {}
                    station_info = row.find_element(By.XPATH, ".//div[contains(@class, '_kZZF')]")
                    stop_info['station_name'] = station_info.find_element(By.XPATH, ".//span[contains(@class, '_Hjc4')]").text
                    stop_info['station_code'] = station_info.find_element(By.XPATH, ".//span[contains(@class, 'LlBCs')]").text.strip("()")

                    time_elements = row.find_elements(By.XPATH, ".//div[contains(@class, 'brNEO')]")
                    if len(time_elements) >= 3:
                        stop_info['arrival_time'] = time_elements[0].text
                        stop_info['halt_duration'] = time_elements[1].text
                        stop_info['departure_time'] = time_elements[2].text

                    if "You are boarding here" in row.text:
                        stop_info['is_boarding_point'] = True
                    if "You are droppping off here" in row.text:
                        stop_info['is_dropping_point'] = True

                    stops.append(stop_info)
                except StaleElementReferenceException:
                    logger.warning("Stale element encountered, skipping this stop")
                except Exception as e:
                    logger.error(f"Error extracting stop info: {str(e)}")

            train_info['stops'] = stops
            train_routes.append(train_info)

            # Close the route information modal
            close_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//button[@aria-label='Close']"))
            )
            driver.execute_script("arguments[0].click();", close_button)

            # Wait for the modal to close
            WebDriverWait(driver, 10).until(
                EC.invisibility_of_element_located((By.XPATH, "//div[contains(@class, 'QMO26')]"))
            )
            
            # If we found our target train, we can stop here
            if target_train_number and train_info['number'] == target_train_number:
                break

        except TimeoutException:
            logger.warning(f"Timed out waiting for route information for train {train_info.get('number', 'unknown')}")
        except Exception as e:
            logger.error(f"Error extracting route for train {train_info.get('number', 'unknown')}: {str(e)}")

    return train_routes

def get_thread_driver():
    """Get or create a thread-specific browser instance"""
    thread_id = threading.get_ident()
    
    with browser_lock:
        if thread_id not in browser_instances:
            options = Options()
            options.add_argument("--headless")
            options.add_argument("--disable-gpu")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument(f"user-agent={get_random_user_agent()}")
            options.add_experimental_option("excludeSwitches", ["enable-automation"])
            options.add_experimental_option('useAutomationExtension', False)
            
            driver = webdriver.Chrome(options=options)
            driver.set_page_load_timeout(30)
            driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
                "source": """
                    Object.defineProperty(navigator, 'webdriver', {
                        get: () => undefined
                    })
                """
            })
            
            browser_instances[thread_id] = driver
            logger.info(f"Created new browser instance for thread {thread_id}")
            
        return browser_instances[thread_id]

def cleanup_browsers():
    """Cleanup all browser instances"""
    with browser_lock:
        for thread_id, driver in browser_instances.items():
            try:
                logger.info(f"Cleaning up browser instance for thread {thread_id}")
                driver.quit()
            except:
                pass
        browser_instances.clear()

# Register cleanup function
atexit.register(cleanup_browsers)

def scrape_train_routes(from_station, to_station, date, target_train_number=None, max_retries=3):
    """Scrape train routes using thread-specific browser"""
    url = f"https://tickets.paytm.com/trains/searchTrains/{from_station}/{to_station}/{date}"
    
    for attempt in range(max_retries):
        try:
            driver = get_thread_driver()
            logger.info(f"Attempt {attempt + 1}: Navigating to URL: {url}" + 
                       (f" for train {target_train_number}" if target_train_number else ""))
            
            driver.get(url)
            time.sleep(random.uniform(2, 4))
            
            # Scroll down the page
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight / 2);")
            time.sleep(random.uniform(1, 2))
            
            # Wait for the train list to load
            wait_for_element(driver, By.CLASS_NAME, "Gwgxn", timeout=60)
            
            # Extract train routes
            train_routes = extract_train_routes(driver, target_train_number)
            
            return train_routes
            
        except Exception as e:
            logger.error(f"Error on attempt {attempt + 1}: {str(e)}")
            
            # Only create new browser instance on fatal errors
            if "invalid session id" in str(e).lower() or "no such session" in str(e).lower():
                with browser_lock:
                    thread_id = threading.get_ident()
                    if thread_id in browser_instances:
                        try:
                            browser_instances[thread_id].quit()
                        except:
                            pass
                        del browser_instances[thread_id]
            
            if attempt == max_retries - 1:
                return None
            
            time.sleep(random.uniform(3, 5))
    
    return None

def extract_train_stops(driver, target_train_number=None):
    """Extract just the stops information"""
    try:
        train_elements = driver.find_elements(By.CLASS_NAME, "Gwgxn")
        logger.info(f"Found {len(train_elements)} train elements")
        
        for train in train_elements[:3]:  # Process only first 3 trains for speed
            try:
                train_number = train.find_element(By.CLASS_NAME, "train-number").text.strip()
                
                # Clean the train number (remove any extra text)
                train_number = train_number.split('(')[-1].replace(')', '').strip()
                
                if target_train_number and train_number != target_train_number:
                    continue
                    
                # Check if we already have this train's stops
                if stops_store.has_stops(train_number):
                    logger.info(f"✓ Using cached stops for train {train_number}")
                    return stops_store.get_stops(train_number)
                
                logger.info(f"Extracting stops for train {train_number}")
                
                # Click on the train to expand details
                train.click()
                time.sleep(1)
                
                stops = []
                stop_elements = train.find_elements(By.CLASS_NAME, "stop-info")
                
                for stop in stop_elements:
                    try:
                        stop_data = {
                            'station_name': stop.find_element(By.CLASS_NAME, "station-name").text.strip(),
                            'station_code': stop.find_element(By.CLASS_NAME, "station-code").text.strip(),
                            'arrival_time': stop.find_element(By.CLASS_NAME, "arrival-time").text.strip(),
                            'departure_time': stop.find_element(By.CLASS_NAME, "departure-time").text.strip(),
                            'halt_duration': stop.find_element(By.CLASS_NAME, "halt-duration").text.strip()
                        }
                        stops.append(stop_data)
                    except Exception as e:
                        logger.warning(f"Error extracting stop data: {str(e)}")
                        continue
                
                if stops:
                    logger.info(f"✓ Storing {len(stops)} stops for train {train_number}")
                    stops_store.add_stops(train_number, stops)
                    return stops
                    
                if target_train_number:
                    break
                    
            except Exception as e:
                logger.error(f"Error processing train element: {str(e)}")
                continue
                
    except Exception as e:
        logger.error(f"Error in extract_train_stops: {str(e)}")
        return None

def get_train_stops(train_number: str, from_station: str, to_station: str, date: str) -> Optional[List[dict]]:
    """Get train stops using thread-specific browser"""
    train_number = train_number.split('(')[-1].replace(')', '').strip()
    
    # Check cache first
    stored_stops = stops_store.get_stops(train_number)
    if stored_stops:
        logger.info(f"✓ CACHE HIT: Using cached stops for train {train_number}")
        return stored_stops

    logger.info(f"✗ CACHE MISS: Need to scrape stops for train {train_number}")
    try:
        driver = get_thread_driver()
        url = f"https://tickets.paytm.com/trains/searchTrains/{from_station}/{to_station}/{date}"
        
        driver.get(url)
        time.sleep(random.uniform(1, 2))
        
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "Gwgxn"))
        )
        
        stops = extract_train_stops(driver, train_number)
        if stops:
            logger.info(f"✓ Successfully scraped and stored stops for train {train_number}")
            return stops
        else:
            logger.warning(f"Failed to scrape stops for train {train_number}")
            return None
            
    except Exception as e:
        logger.error(f"Error scraping train {train_number}: {str(e)}")
        return None

# Example usage
if __name__ == "__main__":
    # Example: Get stops for a train
    stops = get_train_stops("12345", "NDLS", "HWH", "20240321")
    if stops:
        print(f"Stops for train 12345:")
        for stop in stops:
            print(f"  {stop['station_name']} ({stop['station_code']})")
            print(f"    Arrival: {stop['arrival_time']}")
            print(f"    Departure: {stop['departure_time']}")
            print(f"    Halt: {stop['halt_duration']}")