import json
import os
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class TrainStopsStore:
    def __init__(self, store_file: str = "train_stops.json"):
        self.store_file = store_file
        self.stops: Dict[str, List[dict]] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.load_stops()

    def load_stops(self):
        """Load stored train stops from JSON file"""
        if os.path.exists(self.store_file):
            try:
                with open(self.store_file, 'r') as f:
                    self.stops = json.load(f)
                logger.info(f"✓ Loaded {len(self.stops)} train stops from cache file")
            except Exception as e:
                logger.error(f"Error loading cache file: {str(e)}")
                self.stops = {}
        else:
            logger.info("No cache file found, starting with empty cache")
            self.stops = {}

    def save_stops(self):
        """Save train stops to JSON file"""
        try:
            with open(self.store_file, 'w') as f:
                json.dump(self.stops, f, indent=2)
            logger.info(f"✓ Saved {len(self.stops)} train stops to cache file")
        except Exception as e:
            logger.error(f"Error saving cache file: {str(e)}")

    def get_stops(self, train_number: str) -> Optional[List[dict]]:
        """Get stops for a train number"""
        stops = self.stops.get(train_number)
        if stops:
            self.cache_hits += 1
            logger.info(f"✓ Cache hit for train {train_number} (hits: {self.cache_hits}, misses: {self.cache_misses})")
        else:
            self.cache_misses += 1
            logger.info(f"✗ Cache miss for train {train_number} (hits: {self.cache_hits}, misses: {self.cache_misses})")
        return stops

    def add_stops(self, train_number: str, stops: List[dict]):
        """Add new train stops"""
        self.stops[train_number] = stops
        logger.info(f"✓ Added {len(stops)} stops for train {train_number} to cache")
        self.save_stops()

    def has_stops(self, train_number: str) -> bool:
        """Check if stops exist for train"""
        exists = train_number in self.stops
        if exists:
            logger.debug(f"✓ Found train {train_number} in cache")
        return exists

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'total_trains': len(self.stops)
        }

    def update_stops(self, train_number: str, stops: List[dict]):
        """Update existing train stops"""
        if train_number in self.stops:
            self.stops[train_number] = stops
            self.save_stops()
            return True
        return False

    def clear_stops(self, train_number: str = None):
        """Clear stops for one train or all trains"""
        if train_number:
            self.stops.pop(train_number, None)
        else:
            self.stops.clear()
        self.save_stops() 