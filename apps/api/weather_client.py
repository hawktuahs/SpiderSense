"""
Weather data client for SmartSense API.

Provides weather forecasts for energy load modeling with:
- OpenWeatherMap API integration
- Local caching for offline demo mode
- Indian city support with timezone handling
"""

import os
import json
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class WeatherClient:
    """Weather data client with caching and offline support."""
    
    def __init__(self, api_key: Optional[str] = None, cache_dir: str = "data/cache"):
        self.api_key = api_key or os.getenv("OWM_API_KEY")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.base_url = "http://api.openweathermap.org/data/2.5"
        
        # Indian cities with coordinates for demo
        self.indian_cities = {
            "mumbai": {"lat": 19.0760, "lon": 72.8777},
            "delhi": {"lat": 28.7041, "lon": 77.1025},
            "bangalore": {"lat": 12.9716, "lon": 77.5946},
            "chennai": {"lat": 13.0827, "lon": 80.2707},
            "kolkata": {"lat": 22.5726, "lon": 88.3639},
            "hyderabad": {"lat": 17.3850, "lon": 78.4867},
            "pune": {"lat": 18.5204, "lon": 73.8567},
            "ahmedabad": {"lat": 23.0225, "lon": 72.5714},
            "jaipur": {"lat": 26.9124, "lon": 75.7873},
            "lucknow": {"lat": 26.8467, "lon": 80.9462}
        }
        
        # Initialize demo cache
        asyncio.create_task(self._initialize_demo_cache())
    
    async def get_forecast(
        self,
        city: str,
        hours: int,
        freq: str = "H"
    ) -> Optional[pd.DataFrame]:
        """
        Get weather forecast for specified city and duration.
        
        Args:
            city: City name (Indian cities supported)
            hours: Forecast horizon in hours
            freq: Frequency (H for hourly, D for daily)
            
        Returns:
            DataFrame with weather variables or None if unavailable
        """
        city_lower = city.lower()
        
        # Try API first if key available
        if self.api_key:
            try:
                forecast_data = await self._fetch_api_forecast(city_lower, hours, freq)
                if forecast_data is not None:
                    return forecast_data
            except Exception as e:
                logger.warning(f"API forecast failed for {city}: {e}")
        
        # Fallback to cached/demo data
        return await self._get_cached_forecast(city_lower, hours, freq)
    
    async def get_current_weather(self, city: str) -> Optional[Dict[str, float]]:
        """Get current weather conditions."""
        city_lower = city.lower()
        
        if self.api_key:
            try:
                return await self._fetch_current_weather(city_lower)
            except Exception as e:
                logger.warning(f"Current weather API failed for {city}: {e}")
        
        # Return demo current weather
        return self._generate_demo_current_weather(city_lower)
    
    async def _fetch_api_forecast(
        self,
        city: str,
        hours: int,
        freq: str
    ) -> Optional[pd.DataFrame]:
        """Fetch forecast from OpenWeatherMap API."""
        if city not in self.indian_cities:
            logger.warning(f"City {city} not in supported list")
            return None
        
        coords = self.indian_cities[city]
        
        # Use 5-day forecast endpoint (3-hour intervals)
        url = f"{self.base_url}/forecast"
        params = {
            "lat": coords["lat"],
            "lon": coords["lon"],
            "appid": self.api_key,
            "units": "metric"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    logger.error(f"Weather API error: {response.status}")
                    return None
                
                data = await response.json()
                return self._process_api_forecast(data, hours, freq)
    
    async def _fetch_current_weather(self, city: str) -> Optional[Dict[str, float]]:
        """Fetch current weather from API."""
        if city not in self.indian_cities:
            return None
        
        coords = self.indian_cities[city]
        url = f"{self.base_url}/weather"
        params = {
            "lat": coords["lat"],
            "lon": coords["lon"],
            "appid": self.api_key,
            "units": "metric"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    return None
                
                data = await response.json()
                return {
                    "temperature": data["main"]["temp"],
                    "humidity": data["main"]["humidity"],
                    "pressure": data["main"]["pressure"],
                    "wind_speed": data["wind"]["speed"]
                }
    
    def _process_api_forecast(
        self,
        api_data: Dict,
        hours: int,
        freq: str
    ) -> pd.DataFrame:
        """Process OpenWeatherMap API response into DataFrame."""
        forecasts = []
        
        for item in api_data["list"][:hours//3]:  # API provides 3-hour intervals
            timestamp = datetime.fromtimestamp(item["dt"])
            
            forecast_point = {
                "timestamp": timestamp,
                "temperature": item["main"]["temp"],
                "humidity": item["main"]["humidity"],
                "pressure": item["main"]["pressure"],
                "wind_speed": item["wind"]["speed"]
            }
            
            forecasts.append(forecast_point)
        
        df = pd.DataFrame(forecasts)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
        
        # Interpolate to hourly if needed
        if freq == "H" and len(df) > 1:
            hourly_index = pd.date_range(
                start=df.index[0],
                periods=hours,
                freq="H"
            )
            df = df.reindex(hourly_index, method="nearest")
            df = df.interpolate(method="linear")
        
        return df
    
    async def _get_cached_forecast(
        self,
        city: str,
        hours: int,
        freq: str
    ) -> Optional[pd.DataFrame]:
        """Get forecast from cache or generate demo data."""
        cache_file = self.cache_dir / f"weather_{city}.json"
        
        # Try loading from cache
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                
                # Check if cache is recent (within 6 hours)
                cache_time = datetime.fromisoformat(cached_data["timestamp"])
                if datetime.utcnow() - cache_time < timedelta(hours=6):
                    return self._process_cached_forecast(cached_data, hours, freq)
            except Exception as e:
                logger.warning(f"Cache read failed for {city}: {e}")
        
        # Generate demo forecast
        return self._generate_demo_forecast(city, hours, freq)
    
    def _process_cached_forecast(
        self,
        cached_data: Dict,
        hours: int,
        freq: str
    ) -> pd.DataFrame:
        """Process cached weather data."""
        forecasts = cached_data["forecasts"][:hours]
        
        df = pd.DataFrame(forecasts)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
        
        return df
    
    def _generate_demo_forecast(
        self,
        city: str,
        hours: int,
        freq: str
    ) -> pd.DataFrame:
        """Generate realistic demo weather forecast."""
        if city not in self.indian_cities:
            city = "mumbai"  # Default fallback
        
        # Base weather patterns for Indian cities
        base_temps = {
            "mumbai": 28, "delhi": 25, "bangalore": 22, "chennai": 30,
            "kolkata": 27, "hyderabad": 26, "pune": 24, "ahmedabad": 29,
            "jaipur": 26, "lucknow": 25
        }
        
        base_humidity = {
            "mumbai": 75, "delhi": 60, "bangalore": 65, "chennai": 80,
            "kolkata": 70, "hyderabad": 55, "pune": 60, "ahmedabad": 50,
            "jaipur": 45, "lucknow": 65
        }
        
        base_temp = base_temps.get(city, 26)
        base_humid = base_humidity.get(city, 65)
        
        # Generate timestamps
        start_time = datetime.utcnow()
        if freq == "H":
            timestamps = pd.date_range(start=start_time, periods=hours, freq="H")
        else:
            timestamps = pd.date_range(start=start_time, periods=hours, freq="D")
        
        # Generate realistic patterns
        np.random.seed(42)  # Reproducible demo data
        
        # Temperature with diurnal cycle
        hour_of_day = np.array([ts.hour for ts in timestamps])
        temp_cycle = 5 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
        temperature = base_temp + temp_cycle + np.random.normal(0, 2, len(timestamps))
        
        # Humidity (inverse correlation with temperature)
        humidity = base_humid - 0.5 * temp_cycle + np.random.normal(0, 5, len(timestamps))
        humidity = np.clip(humidity, 20, 95)
        
        # Pressure (relatively stable with small variations)
        pressure = 1013 + np.random.normal(0, 5, len(timestamps))
        
        # Wind speed (random with some correlation to time of day)
        wind_speed = 3 + 2 * np.sin(2 * np.pi * hour_of_day / 24) + np.random.exponential(2, len(timestamps))
        wind_speed = np.clip(wind_speed, 0, 15)
        
        return pd.DataFrame({
            "temperature": temperature,
            "humidity": humidity,
            "pressure": pressure,
            "wind_speed": wind_speed
        }, index=timestamps)
    
    def _generate_demo_current_weather(self, city: str) -> Dict[str, float]:
        """Generate demo current weather."""
        base_temps = {
            "mumbai": 28, "delhi": 25, "bangalore": 22, "chennai": 30,
            "kolkata": 27, "hyderabad": 26, "pune": 24, "ahmedabad": 29,
            "jaipur": 26, "lucknow": 25
        }
        
        base_humidity = {
            "mumbai": 75, "delhi": 60, "bangalore": 65, "chennai": 80,
            "kolkata": 70, "hyderabad": 55, "pune": 60, "ahmedabad": 50,
            "jaipur": 45, "lucknow": 65
        }
        
        current_hour = datetime.now().hour
        temp_adjustment = 3 * np.sin(2 * np.pi * (current_hour - 6) / 24)
        
        return {
            "temperature": base_temps.get(city, 26) + temp_adjustment,
            "humidity": base_humidity.get(city, 65) - 0.3 * temp_adjustment,
            "pressure": 1013.0,
            "wind_speed": 5.0
        }
    
    async def _initialize_demo_cache(self):
        """Initialize demo weather cache for offline mode."""
        try:
            for city in self.indian_cities.keys():
                cache_file = self.cache_dir / f"weather_{city}.json"
                
                if not cache_file.exists():
                    # Generate and cache demo data
                    demo_forecast = self._generate_demo_forecast(city, 120, "H")  # 5 days
                    
                    cache_data = {
                        "city": city,
                        "timestamp": datetime.utcnow().isoformat(),
                        "forecasts": [
                            {
                                "timestamp": ts.isoformat(),
                                "temperature": row["temperature"],
                                "humidity": row["humidity"],
                                "pressure": row["pressure"],
                                "wind_speed": row["wind_speed"]
                            }
                            for ts, row in demo_forecast.iterrows()
                        ]
                    }
                    
                    with open(cache_file, 'w') as f:
                        json.dump(cache_data, f, indent=2)
                    
                    logger.info(f"Initialized demo weather cache for {city}")
        
        except Exception as e:
            logger.warning(f"Demo cache initialization failed: {e}")
    
    def get_supported_cities(self) -> List[str]:
        """Get list of supported Indian cities."""
        return list(self.indian_cities.keys())
