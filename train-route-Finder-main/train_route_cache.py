import json
import os
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

class TrainRouteCache:
    def __init__(self, cache_file: str = "train_routes_cache.json"):
        self.cache_file = cache_file
        self.routes: Dict[str, dict] = {}
        self.last_updated: Dict[str, str] = {}
        self.load_cache()

    def load_cache(self):
        """Load cached routes from file"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    self.routes = data.get('routes', {})
                    self.last_updated = data.get('last_updated', {})
        except Exception as e:
            logging.error(f"Error loading cache: {e}")
            self.routes = {}
            self.last_updated = {}

    def save_cache(self):
        """Save routes to cache file"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump({
                    'routes': self.routes,
                    'last_updated': self.last_updated
                }, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving cache: {e}")

    def get_route(self, train_number: str) -> Optional[List[dict]]:
        """Get route from cache if available"""
        if train_number in self.routes:
            # Check if cache is not too old (e.g., 30 days)
            last_updated = datetime.strptime(
                self.last_updated.get(train_number, '2000-01-01'),
                '%Y-%m-%d'
            )
            if datetime.now() - last_updated < timedelta(days=30):
                return self.routes[train_number]
        return None

    def add_route(self, train_number: str, stops: List[dict]):
        """Add or update route in cache"""
        self.routes[train_number] = stops
        self.last_updated[train_number] = datetime.now().strftime('%Y-%m-%d')
        self.save_cache()

    def is_route_cached(self, train_number: str) -> bool:
        """Check if route exists in cache and is not expired"""
        return self.get_route(train_number) is not None

# Example cache structure:
"""
{
    "routes": {
        "12345": [
            {
                "station_name": "New Delhi",
                "station_code": "NDLS",
                "arrival_time": "00:00",
                "departure_time": "08:00",
                "halt_duration": "00:00"
            },
            {
                "station_name": "Kanpur Central",
                "station_code": "CNB",
                "arrival_time": "13:45",
                "departure_time": "13:50",
                "halt_duration": "00:05"
            }
        ]
    },
    "last_updated": {
        "12345": "2024-03-20"
    }
}
"""