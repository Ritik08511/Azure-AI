from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time
import logging
import random
import threading
from threading import local
import atexit
from typing import Dict
from threading import Lock

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def extract_train_data(driver, target_train_number=None):
    """
    Extract train data, optionally filtering for a specific train number.
    """
    train_data = []
    train_elements = driver.find_elements(By.CLASS_NAME, "Gwgxn")
    thread_name = threading.current_thread().name
    logger.info(f"Thread {thread_name}: Found {len(train_elements)} train elements" + 
                (f", filtering for train {target_train_number}" if target_train_number else ""))

    for train_element in train_elements:
        train_info = {}
        
        try:
            name_number_element = train_element.find_element(By.CLASS_NAME, "k9j0o")
            train_info['name'] = name_number_element.find_element(By.TAG_NAME, "h1").text
            train_info['number'] = name_number_element.find_element(By.CLASS_NAME, "qW4yv").text.strip("()")
            
            # Skip if not the target train
            if target_train_number and train_info['number'] != target_train_number:
                continue
                
            logger.info(f"Thread {thread_name}: Extracting data for train: {train_info['name']} ({train_info['number']})")
        except NoSuchElementException as e:
            logger.error(f"Thread {thread_name}: Error extracting train name/number: {str(e)}")
            continue

        try:
            time_elements = train_element.find_elements(By.CLASS_NAME, "nnGXi")
            train_info['departure_time'] = time_elements[0].text
            train_info['arrival_time'] = time_elements[1].text
            logger.debug(f"Thread {thread_name}: Found times for {train_info['number']}: {train_info['departure_time']} - {train_info['arrival_time']}")
        except IndexError:
            logger.warning(f"Thread {thread_name}: Could not find departure or arrival times for train {train_info['number']}")
            train_info['departure_time'] = "N/A"
            train_info['arrival_time'] = "N/A"

        try:
            duration_element = train_element.find_element(By.CLASS_NAME, "GVfQw")
            train_info['duration'] = duration_element.text
        except NoSuchElementException:
            logger.warning(f"Thread {thread_name}: Could not find journey duration for train {train_info['number']}")
            train_info['duration'] = "N/A"

        try:
            class_containers = train_element.find_elements(By.CLASS_NAME, "PrZHl")
            classes_and_availability = []
            
            for i in range(0, len(class_containers), 2):
                class_info = {}
                
                first_container = class_containers[i]
                try:
                    class_type_element = first_container.find_element(By.CLASS_NAME, "bGfcC")
                    class_info['type'] = class_type_element.text
                    availability_element = first_container.find_element(By.CLASS_NAME, "envfU")
                    class_info['availability'] = availability_element.text
                    logger.debug(f"Thread {thread_name}: Found class {class_info['type']} with availability {class_info['availability']}")
                except NoSuchElementException:
                    continue  
                
                if i + 1 < len(class_containers):
                    second_container = class_containers[i + 1]
                    try:
                        price_element = second_container.find_element(By.CLASS_NAME, "SHHaW")
                        price_text = price_element.text.split('\n')[0]
                        class_info['price'] = price_text
                    except NoSuchElementException:
                        class_info['price'] = "N/A"
                else:
                    class_info['price'] = "N/A"
                
                classes_and_availability.append(class_info)
            
            train_info['classes_and_availability'] = classes_and_availability
            
        except Exception as e:
            logger.warning(f"Thread {thread_name}: Could not find class and availability information for train {train_info['number']}: {str(e)}")
            train_info['classes_and_availability'] = []

        try:
            chance_elements = train_element.find_elements(By.CLASS_NAME, "Ob72l")
            train_info['confirmation_chances'] = [element.text for element in chance_elements]
        except NoSuchElementException:
            logger.warning(f"Thread {thread_name}: Could not find confirmation chances for train {train_info['number']}")
            train_info['confirmation_chances'] = []

        try:
            station_elements = train_element.find_elements(By.CLASS_NAME, "pYpdU")
            train_info['from_station'] = station_elements[0].text
            train_info['to_station'] = station_elements[1].text
        except IndexError:
            logger.warning(f"Thread {thread_name}: Could not find station information for train {train_info['number']}")
            train_info['from_station'] = "N/A"
            train_info['to_station'] = "N/A"

        train_data.append(train_info)
        
        # If we found our target train, we can stop here
        if target_train_number and train_info['number'] == target_train_number:
            break

    return train_data

def scrape_train_data(from_station, to_station, date, target_train_number=None, max_retries=3):
    """Scrape train data using thread-specific browser"""
    url = f"https://tickets.paytm.com/trains/searchTrains/{from_station}/{to_station}/{date}"
    thread_name = threading.current_thread().name
    
    for attempt in range(max_retries):
        try:
            driver = get_thread_driver()
            logger.info(f"Thread {thread_name}: Attempt {attempt + 1}: Navigating to URL: {url}" +
                       (f" for train {target_train_number}" if target_train_number else ""))
            
            driver.get(url)
            time.sleep(random.uniform(2, 4))
            
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight / 2);")
            time.sleep(random.uniform(1, 2))
            
            WebDriverWait(driver, 60).until(
                EC.presence_of_element_located((By.CLASS_NAME, "Gwgxn"))
            )
            
            train_data = extract_train_data(driver, target_train_number)
            return train_data
            
        except Exception as e:
            logger.error(f"Thread {thread_name}: Error on attempt {attempt + 1}: {str(e)}")
            
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

if __name__ == "__main__":
    data = scrape_train_data("NDLS_Delhi", "HWH_Howrah", "20250214")
    if data:
        for train in data:
            print(f"\nTrain: {train['name']} ({train['number']})")
            print(f"From: {train['from_station']} To: {train['to_station']}")
            print(f"Departure: {train['departure_time']}, Arrival: {train['arrival_time']}")
            print(f"Duration: {train['duration']}")
            print("\nClass and Seat Availability:")
            for class_info in train['classes_and_availability']:
                print(f"  {class_info['type']}")
                print(f"  - Availability: {class_info['availability']}")
                print(f"  - Price: {class_info['price']}")
                print("  ---")
            print(f"Confirmation Chances: {', '.join(train['confirmation_chances'])}")
            print("-" * 50)
    else:
        print("Failed to retrieve train data.")