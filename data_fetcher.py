import os
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging

#CONFIGURATION 
API_KEY = "your_mapbox_api_key_here"  # Replace with your Mapbox/Google token
BASE_DATA_PATH = "data/train(1).csv"
SAVE_DIR = "data/property_images/"
LOG_FILE = "logs/download_errors.log"
ZOOM_LEVEL = 17
IMAGE_SIZE = "400x400"
THREADS = 10
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)
logging.basicConfig(filename=LOG_FILE, level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def download_satellite_image(row):
    """
    Fetches a single satellite image for a given property row.
    """
    house_id = row['id']
    lat = row['lat']
    lon = row['long']
    
    file_path = os.path.join(SAVE_DIR, f"{house_id}.jpg")
    
    if os.path.exists(file_path):
        return "Exists"

    # Mapbox Static Image URL 
    url = f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/{lon},{lat},{ZOOM_LEVEL},0/{IMAGE_SIZE}?access_token={API_KEY}"

    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
            return "Success"
        else:
            logging.error(f"ID {house_id}: API Error {response.status_code} - {response.text}")
            return "Error"
    except Exception as e:
        logging.error(f"ID {house_id}: Exception {str(e)}")
        return "Exception"

def main():
    #Load Data
    print(f"[*] Loading data from {BASE_DATA_PATH}...")
    df = pd.read_csv(BASE_DATA_PATH)
    
    #Convert to list
    records = df.to_dict('records')
    total_images = len(records)
    
    print(f"[*] Starting download of {total_images} images using {THREADS} threads...")
    
    results = {"Success": 0, "Exists": 0, "Error": 0, "Exception": 0}
    
    with ThreadPoolExecutor(max_workers=THREADS) as executor:
    
        futures = {executor.submit(download_satellite_image, row): row for row in records}
        
        # Monitor with progress bar
        for future in tqdm(as_completed(futures), total=total_images, desc="Downloading Tiles"):
            res = future.result()
            results[res] += 1

    #Final Summary
    print("\n--- Download Summary ---")
    for key, value in results.items():
        print(f"{key}: {value}")
    print(f"Check {LOG_FILE} for any failures.")

if __name__ == "__main__":
    main()