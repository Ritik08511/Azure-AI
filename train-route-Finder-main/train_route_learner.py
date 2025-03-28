import torch
import torch.nn as nn
import torch.optim as optim
import json
import random
from collections import defaultdict
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor
import os
from typing import List

class StationRouteNetwork(nn.Module):
    def __init__(self, num_stations):
        super().__init__()
        self.network = nn.Sequential(
            # Input features: [origin_freq, dest_freq, intermediate_freq, common_trains]
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """Forward pass through the network"""
        return self.network(x)

class FastTrainRoutePredictor:
    def __init__(self, train_stops_file, station_names_file=None):
        self.model_file = 'route_predictor.pt'
        
        # Load train stops data
        with open(train_stops_file, 'r') as f:
            self.train_stops = json.load(f)
        
        # Load station names or create default mapping    
        self.station_names = {}
        if station_names_file and os.path.exists(station_names_file):
            with open(station_names_file, 'r') as f:
                self.station_names = json.load(f)
        else:
            # Extract station names from train stops if available
            for train_number, stops in self.train_stops.items():
                for stop in stops:
                    if 'station_code' in stop and 'station_name' in stop:
                        self.station_names[stop['station_code']] = stop['station_name']
            
            # Save extracted names for future use
            if self.station_names:
                with open('station_names.json', 'w') as f:
                    json.dump(self.station_names, f, indent=2)
            
        # Initialize station statistics
        self.station_stats = defaultdict(lambda: {
            'frequency': 0,
            'connections': defaultdict(set),
            'success_count': 0,
            'route_successes': defaultdict(int)
        })
        
        # Process train stops data once
        self._process_train_data()
        
        # Load successful routes data immediately (without training)
        self._quick_load_routes()
        
        # Initialize model and training in background
        self.train_executor = ThreadPoolExecutor(max_workers=1)
        self.train_executor.submit(self._background_model_init)
        
        self.cache_lock = threading.Lock()

    def _process_train_data(self):
        """Process train stops to build station statistics"""
        for train_number, stops in self.train_stops.items():
            # Count station frequencies
            for stop in stops:
                station_code = stop['station_code']
                self.station_stats[station_code]['frequency'] += 1
                
                # Store station name if available
                if 'station_name' in stop and station_code not in self.station_names:
                    self.station_names[station_code] = stop['station_name']
                
            # Build connections
            for i, stop1 in enumerate(stops):
                for stop2 in stops[i+1:]:
                    station1 = stop1['station_code']
                    station2 = stop2['station_code']
                    self.station_stats[station1]['connections'][station2].add(train_number)
                    self.station_stats[station2]['connections'][station1].add(train_number)

    def _quick_load_routes(self):
        """Quickly load route statistics without training"""
        try:
            if os.path.exists('successful_routes.json'):
                with open('successful_routes.json', 'r') as f:
                    routes_data = json.load(f)
                    for route in routes_data:
                        origin = route['origin']
                        dest = route['destination']
                        intermediate = route['intermediate']
                        self.station_stats[intermediate]['success_count'] += 1
                        self.station_stats[intermediate]['route_successes'][(origin, dest)] += 1
                print(f"Loaded {len(routes_data)} route statistics")
        except Exception as e:
            print(f"Error quick loading routes: {e}")

    def _background_model_init(self):
        """Initialize and train model in background"""
        try:
            # Initialize model
            self.model = StationRouteNetwork(len(self.station_stats))
            self.optimizer = optim.Adam(self.model.parameters())
            self.criterion = nn.BCELoss()
            
            # Load existing model if available
            if os.path.exists(self.model_file):
                checkpoint = torch.load(self.model_file)
                self.model.load_state_dict(checkpoint['model_state'])
                if 'optimizer_state' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            
            # Train on existing routes
            if os.path.exists('successful_routes.json'):
                with open('successful_routes.json', 'r') as f:
                    routes_data = json.load(f)
                    for route in routes_data:
                        self.train_on_route(
                            route['origin'], 
                            route['destination'], 
                            route['intermediate']
                        )
                print("Background model training completed")
        except Exception as e:
            print(f"Error in background initialization: {e}")

    def create_features(self, origin, intermediate, destination):
        """Enhanced feature creation for better predictions"""
        max_freq = max(s['frequency'] for s in self.station_stats.values())
        
        # Basic frequency features
        origin_freq = self.station_stats[origin]['frequency'] / max_freq
        dest_freq = self.station_stats[destination]['frequency'] / max_freq
        inter_freq = self.station_stats[intermediate]['frequency'] / max_freq
        
        # Enhanced connectivity feature
        common_trains = len(
            self.station_stats[origin]['connections'][intermediate] &
            self.station_stats[intermediate]['connections'][destination]
        )
        max_common = max(
            len(connections) 
            for stat in self.station_stats.values() 
            for connections in stat['connections'].values()
        ) or 1  # Avoid division by zero
        
        return torch.tensor([
            origin_freq,
            dest_freq,
            inter_freq,
            common_trains / max_common
        ], dtype=torch.float32)

    def get_station_with_name(self, station_code):
        """Format station code with name in route finder format (CODE_StationName)"""
        try:
            if station_code in self.station_names:
                # Get raw station name
                station_name = self.station_names[station_code]
                
                # Format name: remove extra spaces, capitalize words, remove special chars
                words = station_name.strip().split()
                formatted_name = ''.join(word.capitalize() for word in words)
                formatted_name = ''.join(c for c in formatted_name if c.isalnum())
                
                # Return in CODE_StationName format
                return f"{station_code}_{formatted_name}"
            return f"{station_code}_Unknown"
        except Exception as e:
            print(f"Error formatting station name for {station_code}: {e}")
            return f"{station_code}_Unknown"

    def predict_intermediate_stations(self, origin: str, destination: str, top_n: int = 5) -> List[str]:
        """Predict best intermediate stations using both ML and statistics"""
        try:
            predictions = []
            route_key = (origin, destination)
            
            # Create a stable list of stations to avoid dictionary modification issues
            stations_list = list(self.station_stats.keys())
            
            for station in stations_list:
                if station in [origin, destination]:
                    continue
                    
                score = 0.0
                
                # 1. Check success history (highest priority)
                success_count = self.station_stats[station]['route_successes'][route_key]
                if success_count > 0:
                    score = 1.0 + success_count * 0.1
                    predictions.append((station, score))
                    continue
                
                # 2. Check ML prediction if model is available
                if hasattr(self, 'model') and self.model is not None:
                    try:
                        with torch.no_grad():
                            features = self.create_features(origin, station, destination)
                            ml_score = self.model(features).item()
                            score = max(score, ml_score)
                    except Exception as e:
                        print(f"ML prediction error for {station}: {e}")
                
                # 3. Calculate connectivity score
                try:
                    common_trains = len(
                        self.station_stats[origin]['connections'][station] &
                        self.station_stats[station]['connections'][destination]
                    )
                    
                    if common_trains > 0:
                        freq = self.station_stats[station]['frequency']
                        max_freq = max(s['frequency'] for s in self.station_stats.values())
                        connectivity_score = (common_trains / 10) * 0.7 + (freq / max_freq) * 0.3
                        score = max(score, connectivity_score)
                except Exception as e:
                    print(f"Connectivity calculation error for {station}: {e}")
                
                if score > 0:
                    predictions.append((station, score))
            
            # Sort and get top stations
            sorted_stations = sorted(predictions, key=lambda x: x[1], reverse=True)
            result_codes = [station for station, _ in sorted_stations[:top_n]]
            
            # If we need more stations, add major junctions
            if len(result_codes) < top_n:
                remaining = top_n - len(result_codes)
                major_stations = sorted(
                    [(s, self.station_stats[s]['frequency']) 
                     for s in stations_list 
                     if s not in [origin, destination] + result_codes],
                    key=lambda x: x[1],
                    reverse=True
                )
                result_codes.extend([s for s, _ in major_stations[:remaining]])
            
            # Convert station codes to code_name format with no spaces in names
            result = [self.get_station_with_name(station) for station in result_codes]
            
            print(f"Predicted stations for {origin}-{destination}: {result}")
            return result
            
        except Exception as e:
            print(f"Error in predict_intermediate_stations: {e}")
            # Fallback to basic frequency-based prediction
            try:
                major_stations = sorted(
                    [(s, self.station_stats[s]['frequency']) 
                     for s in self.station_stats.keys()
                     if s not in [origin, destination]],
                    key=lambda x: x[1],
                    reverse=True
                )
                result_codes = [s for s, _ in major_stations[:top_n]]
                return [self.get_station_with_name(station) for station in result_codes]
            except:
                return []

    # In the FastTrainRoutePredictor class
    def update_route_async(self, origin, destination, successful_intermediate):
        """Update model and stats in background"""
        def background_update():
            with self.cache_lock:
                # Update statistics immediately
                route_key = (origin, destination)
                self.station_stats[successful_intermediate]['success_count'] += 1
                self.station_stats[successful_intermediate]['route_successes'][route_key] += 1
                
                # Train model if initialized
                if hasattr(self, 'model'):
                    self.train_on_route(origin, destination, successful_intermediate)
                    self.save_model(self.model_file)
                
                # Save the successful routes to JSON file
                self.save_successful_routes('successful_routes.json')
        
        self.train_executor.submit(background_update)

    def train_on_route(self, origin, destination, successful_intermediate):
        """Train model with higher emphasis on successful routes"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Update success statistics
        route_key = (origin, destination)
        self.station_stats[successful_intermediate]['success_count'] += 1
        self.station_stats[successful_intermediate]['route_successes'][route_key] += 1
        
        # Create positive example (successful route)
        pos_features = self.create_features(origin, successful_intermediate, destination)
        pos_output = self.model(pos_features)
        pos_loss = self.criterion(pos_output, torch.tensor([1.0])) * 2.0  # Double weight for positive examples
        
        # Create negative examples (avoid successful intermediates)
        neg_loss = 0
        neg_samples = 3
        for _ in range(neg_samples):
            random_station = random.choice(list(self.station_stats.keys()))
            if (random_station not in [origin, destination, successful_intermediate] and
                self.station_stats[random_station]['route_successes'][route_key] == 0):
                neg_features = self.create_features(origin, random_station, destination)
                neg_output = self.model(neg_features)
                neg_loss += self.criterion(neg_output, torch.tensor([0.0]))
        
        # Combined loss with higher weight on positive examples
        total_loss = (pos_loss * 2 + neg_loss) / (2 + neg_samples)
        total_loss.backward()
        self.optimizer.step()

    def save_model(self, filepath):
        """Save model state"""
        try:
            torch.save({
                'model_state': self.model.state_dict(),
                'optimizer_state': self.optimizer.state_dict()
            }, filepath)
            print("Model saved successfully")
        except Exception as e:
            print(f"Error saving model: {e}")

    def load_model(self, filepath):
        """Load model state"""
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.station_to_idx = checkpoint['station_to_idx']

    def save_successful_routes(self, filepath):
        """Save successful route history"""
        try:
            # First check if file exists and load existing data
            existing_routes = []
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r') as f:
                        existing_routes = json.load(f)
                except json.JSONDecodeError:
                    # If file is corrupted, we'll start fresh
                    existing_routes = []
            
            # Create a dictionary for quick lookup of existing routes
            existing_dict = {}
            for route in existing_routes:
                key = (route['origin'], route['destination'], route['intermediate'])
                existing_dict[key] = route
            
            # Update with current data
            for station in self.station_stats.keys():
                for (origin, dest), count in self.station_stats[station]['route_successes'].items():
                    if count > 0:
                        key = (origin, dest, station)
                        if key in existing_dict:
                            # Update existing entry
                            existing_dict[key]['success_count'] = count
                        else:
                            # Add new entry
                            existing_dict[key] = {
                                'origin': origin,
                                'destination': dest,
                                'intermediate': station,
                                'success_count': count
                            }
            
            # Convert back to list and save
            updated_routes = list(existing_dict.values())
            with open(filepath, 'w') as f:
                json.dump(updated_routes, f, indent=2)
                
            print(f"Saved {len(updated_routes)} routes to {filepath}")
        except Exception as e:
            print(f"Error saving routes: {e}")
            import traceback
            traceback.print_exc()

    def save_station_names(self, filepath='station_names.json'):
        """Save station names mapping"""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.station_names, f, indent=2)
            print(f"Saved {len(self.station_names)} station names to {filepath}")
        except Exception as e:
            print(f"Error saving station names: {e}")

# Example usage
if __name__ == "__main__":
    try:
        # Initialize predictor with station names file if available
        predictor = FastTrainRoutePredictor('train_stops.json', 'station_names.json')
        
        # Get instant predictions
        origin = 'NDLS'
        destination = 'CPR'
        recommended = predictor.predict_intermediate_stations(origin, destination)
        
        print(f"\nRecommended stations between {origin} and {destination}:")
        for i, station_with_name in enumerate(recommended, 1):
            # Split to get station code from CODE_StationName format
            station_code = station_with_name.split('_')[0]
            stats = predictor.station_stats[station_code]
            route_key = (origin, destination)
            success_count = stats['route_successes'][route_key]
            common_trains = len(
                stats['connections'][origin] &
                stats['connections'][destination]
            )
            print(f"{i}. {station_with_name}")
            print(f"   Success Count: {success_count}")
            print(f"   Common Trains: {common_trains}")
            print(f"   Total Frequency: {stats['frequency']}")
        
        # Update in background
        predictor.update_route_async('NDLS', 'CPR', 'CNB')
        
        # Save station names for future use
        predictor.save_station_names()
        
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()