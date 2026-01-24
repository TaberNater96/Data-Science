##########################################################
#       Portland Urban Heat Island Analysis Module       #
#                 By: Elijah Taber                       #
##########################################################

import numpy as np
import pandas as pd
import cv2
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class PortlandHeatAnalyzer:
    """
    Complete analyzer suite for Urban Heat Island (UHI) analysis in Portland/Vancouver metro area.

    This class provides a suite of spatial and temporal analysis methods to examine urban heat
    disparities using VIIRS Land Surface Temperature (LST) data. The analyzer divides the study
    area into distinct zones (urban core, suburban, forest, rural, mountain, and river-adjacent)
    and calculates various heat metrics. The class is designed to work with VIIRS h09v04 tile data 
    covering the Portland region.

    Capabilities:
    -----------------
    - **Zone-based Analysis**: Compare temperatures across different land use types
    - **UHII Calculation**: Quantify Urban Heat Island Intensity (urban vs. rural difference)
    - **Hot Spot Detection**: Identify areas exceeding temperature thresholds
    - **Persistence Tracking**: Find areas that are consistently hot across multiple days
    - **Gradient Analysis**: Examine temperature changes with distance from urban center
    - **Cooling Corridors**: Identify linear cooling features (rivers, green spaces)
    - **Thermal Clustering**: Segment landscape into distinct thermal zones
    - **Heat Wave Detection**: Identify extended periods of high temperature

    Attributes:
    -----------
    tile_shape : tuple of int
        Shape of the LST data array (rows, cols), default (1200, 1200)
    zones : dict of {str: np.ndarray}
        Dictionary of boolean masks defining study zones:
        - 'urban_core': Portland/Vancouver metropolitan core
        - 'suburban': Suburban/residential ring around urban core
        - 'forest_park': Forest Park and western hills
        - 'rural': Eastern agricultural/rural areas
        - 'mountain': Mt. Hood and Cascade foothills
        - 'river_adjacent': Columbia and Willamette river corridors

    Notes:
    ------
    - Input LST data should use NaN for no-data/invalid pixels
    - Zone definitions are approximate and based on geographic knowledge of the Portland area
    - All temperature values are assumed to be in degrees Celsius
    - Spatial resolution is assumed to be approximately 1km (VIIRS LST resolution)
    """
    def __init__(self, tile_shape=(1200, 1200)):
        """
        Initialize analyzer with study area definitions.
        
        Parameters:
        -----------
        tile_shape : tuple
            Shape of the LST data array (rows, cols)
        """
        self.tile_shape = tile_shape
        self.zones = self._define_study_zones()
        
    def _define_study_zones(self) -> Dict[str, np.ndarray]:
        """
        Define spatial zones for Portland area analysis.
        
        Based on h09v04 tile covering Portland area:
        - Northwest: Ocean/Coast
        - Southwest: Urban Portland/Vancouver core
        - Southeast: Rural/Agricultural
        - Northeast: Mt. Hood/Cascade foothills
        
        Returns:
        --------
        Dict of zone masks
        """
        rows, cols = self.tile_shape
        zones = {}
        
        # Urban Core (Portland/Vancouver) - Southwest quadrant
        urban_mask = np.zeros(self.tile_shape, dtype=bool)
        urban_mask[700:1000, 200:500] = True # approximate pixel range based on Portland metropolitan area location
        zones['urban_core'] = urban_mask
        
        # Suburban/Residential - Ring around urban core
        suburban_mask = np.zeros(self.tile_shape, dtype=bool)
        suburban_mask[600:1100, 100:600] = True
        suburban_mask[urban_mask] = False  # Exclude urban core
        zones['suburban'] = suburban_mask
        
        # Forest Park / Western Hills
        forest_mask = np.zeros(self.tile_shape, dtype=bool)
        forest_mask[650:900, 150:350] = True
        forest_mask[urban_mask] = False
        zones['forest_park'] = forest_mask
        
        # Rural/Agricultural - Eastern areas
        rural_mask = np.zeros(self.tile_shape, dtype=bool)
        rural_mask[600:1000, 600:1000] = True
        zones['rural'] = rural_mask
        
        # Mountain/High Elevation - Eastern portions (Mt. Hood area)
        mountain_mask = np.zeros(self.tile_shape, dtype=bool)
        mountain_mask[200:700, 800:1200] = True
        zones['mountain'] = mountain_mask
        
        # River Corridors - Columbia and Willamette (approximate)
        river_mask = np.zeros(self.tile_shape, dtype=bool)
        # Columbia River corridor
        river_mask[750:850, 100:800] = True
        # Willamette River
        river_mask[800:1000, 280:320] = True
        zones['river_adjacent'] = river_mask
        
        return zones
    
    def calculate_uhii(
        self, lst_data: np.ndarray, 
        urban_zone: str = 'urban_core', 
        rural_zone: str = 'rural'
    ) -> float:
        """
        Calculate Urban Heat Island Intensity.
        
        UHII = Mean LST (urban) - Mean LST (rural)
        
        Parameters:
        -----------
        lst_data : np.ndarray
            Land surface temperature data (with NaN for no-data)
        urban_zone : str
            Name of urban zone to use
        rural_zone : str
            Name of rural zone to use
            
        Returns:
        --------
        float : UHII value in degrees C
        """
        urban_mask = self.zones[urban_zone]
        rural_mask = self.zones[rural_zone]
        
        # Extract temperatures for each zone
        urban_temps = lst_data[urban_mask]
        rural_temps = lst_data[rural_mask]
        
        # Remove NaN values
        urban_temps = urban_temps[~np.isnan(urban_temps)]
        rural_temps = rural_temps[~np.isnan(rural_temps)]
        
        if len(urban_temps) == 0 or len(rural_temps) == 0:
            return np.nan
        
        uhii = np.mean(urban_temps) - np.mean(rural_temps)
        return uhii
    
    def zone_statistics(
        self, 
        lst_data: np.ndarray
    ) -> pd.DataFrame:
        """
        Calculate temperature statistics for all zones.
        
        Parameters:
        -----------
        lst_data : np.ndarray
            Land surface temperature data
            
        Returns:
        --------
        pd.DataFrame with columns: zone, mean, std, min, max, p25, p50, p75
        """
        stats = []
        
        for zone_name, zone_mask in self.zones.items():
            zone_temps = lst_data[zone_mask]
            zone_temps = zone_temps[~np.isnan(zone_temps)]
            
            if len(zone_temps) > 0:
                # Make a copy to avoid read-only array issues with numpy 2.0+
                zone_temps_copy = np.array(zone_temps, copy=True)
                
                stats.append({
                    'zone': zone_name,
                    'mean': np.mean(zone_temps_copy),
                    'std': np.std(zone_temps_copy),
                    'min': np.min(zone_temps_copy),
                    'max': np.max(zone_temps_copy),
                    'p25': np.percentile(zone_temps_copy, 25),
                    'p50': np.percentile(zone_temps_copy, 50),
                    'p75': np.percentile(zone_temps_copy, 75),
                    'n_pixels': len(zone_temps_copy)
                })
        
        return pd.DataFrame(stats)
    
    def detect_hot_spots(
        self, 
        lst_data: np.ndarray, 
        threshold_percentile: float = 90
    ) -> np.ndarray:
        """
        Detect hot spots using percentile threshold.
        
        Parameters:
        -----------
        lst_data : np.ndarray
            Temperature data
        threshold_percentile : float
            Percentile to use as threshold (default: 90)
            
        Returns:
        --------
        np.ndarray : Binary mask of hot spots
        """
        valid_data = lst_data[~np.isnan(lst_data)]
        valid_data_copy = np.array(valid_data, copy=True)
        threshold = np.percentile(valid_data_copy, threshold_percentile)
        
        hot_spots = (lst_data >= threshold) & (~np.isnan(lst_data))
        return hot_spots
    
    def persistent_heat_analysis(
        self, 
        lst_series: List[np.ndarray], 
        persistence_days: int = 5
    ) -> Dict:
        """
        Identify areas that are persistently hot.
        
        Parameters:
        -----------
        lst_series : list of np.ndarray
            List of daily LST data
        persistence_days : int
            Minimum number of days to be considered persistent
            
        Returns:
        --------
        Dict with persistent heat metrics
        """
        # Stack all days
        n_days = len(lst_series)
        hot_spot_stack = np.zeros((n_days, *self.tile_shape), dtype=bool)
        
        for i, lst_data in enumerate(lst_series):
            hot_spot_stack[i] = self.detect_hot_spots(lst_data)
        
        # Count how many days each pixel is a hot spot
        hot_spot_count = np.sum(hot_spot_stack, axis=0)
        
        # Persistent hot spots
        persistent_mask = hot_spot_count >= persistence_days
        
        # Calculate percentage of time each pixel is hot
        heat_frequency = (hot_spot_count / n_days) * 100
        
        return {
            'persistent_hot_spots': persistent_mask,
            'heat_frequency': heat_frequency,
            'hot_spot_count': hot_spot_count,
            'n_persistent_pixels': np.sum(persistent_mask),
            'persistence_threshold': persistence_days
        }
    
    def temperature_gradient_analysis(
        self, 
        lst_data: np.ndarray, 
        center_point: Tuple[int, int],
        max_radius: int = 500
    ) -> pd.DataFrame:
        """
        Analyze temperature gradient from a center point (e.g., urban core).
        
        Parameters:
        -----------
        lst_data : np.ndarray
            Temperature data
        center_point : tuple
            (row, col) of center point
        max_radius : int
            Maximum distance to analyze (in pixels)
            
        Returns:
        --------
        pd.DataFrame with distance vs temperature
        """
        rows, cols = self.tile_shape
        
        # Create distance map from center
        y, x = np.ogrid[:rows, :cols]
        distance_map = np.sqrt((x - center_point[1])**2 + (y - center_point[0])**2)
        
        # Bin by distance
        distance_bins = np.arange(0, max_radius, 10)
        binned_temps = []
        
        for i in range(len(distance_bins) - 1):
            mask = (distance_map >= distance_bins[i]) & (distance_map < distance_bins[i+1])
            mask = mask & (~np.isnan(lst_data))
            
            if np.sum(mask) > 0:
                temps = lst_data[mask]
                binned_temps.append({
                    'distance_km': distance_bins[i],  # assuming 1km resolution
                    'mean_temp': np.mean(temps),
                    'std_temp': np.std(temps),
                    'n_pixels': len(temps)
                })
        
        return pd.DataFrame(binned_temps)
    
    def identify_cooling_corridors(
        self, 
        lst_data: np.ndarray, 
        temperature_threshold: float = None
    ) -> np.ndarray:
        """
        Identify cooling corridors (cooler linear features in urban areas).
        
        Uses edge detection and morphological operations.
        
        Parameters:
        -----------
        lst_data : np.ndarray
            Temperature data
        temperature_threshold : float
            Temperature below which to consider "cool" (if None, use 25th percentile)
            
        Returns:
        --------
        np.ndarray : Binary mask of cooling corridors
        """
        # Normalize to 0-255 for OpenCV
        valid_mask = ~np.isnan(lst_data)
        if not np.any(valid_mask):
            return np.zeros(self.tile_shape, dtype=bool)
        
        temp_min = np.nanmin(lst_data)
        temp_max = np.nanmax(lst_data)
        lst_normalized = ((lst_data - temp_min) / (temp_max - temp_min) * 255).astype(np.uint8)
        lst_normalized[~valid_mask] = 0
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(lst_normalized, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)
        
        # Identify cool areas
        if temperature_threshold is None:
            lst_data_copy = np.array(lst_data, copy=True)
            temperature_threshold = np.nanpercentile(lst_data_copy, 25)
        
        cool_areas = (lst_data < temperature_threshold) & valid_mask
        
        # Dilate edges to create corridors
        kernel = np.ones((3, 3), np.uint8)
        corridors = cv2.dilate(edges, kernel, iterations=2)
        cooling_corridors = corridors & cool_areas
        
        return cooling_corridors.astype(bool)
    
    def thermal_clustering(
        self, 
        lst_data: np.ndarray, 
        n_clusters: int = 5
    ) -> np.ndarray:
        """
        Segment temperature data into thermal zones using K-means clustering.
        
        Parameters:
        -----------
        lst_data : np.ndarray
            Temperature data
        n_clusters : int
            Number of thermal zones to identify
            
        Returns:
        --------
        np.ndarray : Cluster labels
        """
        valid_mask = ~np.isnan(lst_data)
        
        # Reshape for clustering
        valid_temps = lst_data[valid_mask].reshape(-1, 1)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(valid_temps)
        
        # Create output array
        cluster_map = np.full(self.tile_shape, -1, dtype=int)
        cluster_map[valid_mask] = labels
        
        return cluster_map, kmeans.cluster_centers_
    
    def heat_wave_detection(
        self, 
        df_stats: pd.DataFrame, 
        threshold_temp: float = 40.0,
        min_duration: int = 3
    ) -> pd.DataFrame:
        """
        Detect heat wave events from time series data.
        
        Parameters:
        -----------
        df_stats : pd.DataFrame
            Time series with columns: date, mean
        threshold_temp : float
            Temperature threshold for heat wave
        min_duration : int
            Minimum consecutive days to be considered heat wave
            
        Returns:
        --------
        pd.DataFrame : Heat wave events
        """
        df = df_stats.copy()
        df['is_hot'] = df['mean'] >= threshold_temp
        
        # Find consecutive hot days
        df['hot_group'] = (df['is_hot'] != df['is_hot'].shift()).cumsum()
        
        heat_waves = []
        for group_id, group in df[df['is_hot']].groupby('hot_group'):
            if len(group) >= min_duration:
                heat_waves.append({
                    'start_date': group['date'].iloc[0],
                    'end_date': group['date'].iloc[-1],
                    'duration_days': len(group),
                    'max_temp': group['mean'].max(),
                    'avg_temp': group['mean'].mean()
                })
        
        return pd.DataFrame(heat_waves)


def create_comparison_visualizations(
    analyzer: PortlandHeatAnalyzer,
    lst_data: np.ndarray,
    output_prefix: str = 'portland_uhi'
):
    """
    Create a full visualization suite for Portland urban heat island analysis.

    Generates a multi-panel figure containing six complementary visualizations:
    study zone overlays, hot spot detection, zone temperature statistics,
    thermal clustering, cooling corridors, and urban-rural temperature gradients.

    Args:
        analyzer: PortlandHeatAnalyzer instance containing zone definitions and analysis methods.
        lst_data: 2D numpy array of land surface temperature data in Celsius.
        output_prefix: Prefix for the output PNG filename. Default is 'portland_uhi'.

    Returns:
        pd.DataFrame: Zone statistics containing mean, std, min, and max temperatures
                      for each defined zone.
    """

    fig = plt.figure(figsize=(20, 12))
    
    # 1. Temperature with zone overlays
    ax1 = plt.subplot(2, 3, 1)
    im1 = ax1.imshow(lst_data, cmap='RdYlBu_r', vmin=10, vmax=50)
    
    # Overlay zones with boundaries
    zone_colors = {
        'urban_core': 'red',
        'suburban': 'orange',
        'forest_park': 'green',
        'rural': 'blue',
        'mountain': 'purple'
    }
    
    # Iterate over each geographic zone (urban_core, suburban, forest_park, etc.)
    for zone_name, zone_mask in analyzer.zones.items():
        # Only process zones that have a defined color
        if zone_name in zone_colors:
            # Use OpenCV to find the boundary contours of the zone mask
            contours = cv2.findContours(zone_mask.astype(np.uint8), 
                                       cv2.RETR_EXTERNAL,          # gets only outer boundaries
                                       cv2.CHAIN_APPROX_SIMPLE)[0] # simplifies contour data
            for contour in contours:
                # Remove extra dimensions from contour array to make it 2D (x,y coordinates)
                contour = contour.squeeze()
                
                # Check if contour has 2 dimensions (meaning it has actual coordinate data) to prevent errors before plotting
                if len(contour.shape) == 2:
                    ax1.plot(
                        contour[:, 0], 
                        contour[:, 1], 
                        color=zone_colors[zone_name], 
                        linewidth=2, 
                        label=zone_name.replace('_', ' ').title(), 
                        alpha=0.7
                    )
    
    ax1.set_title('Study Zones Overlay', fontweight='bold', fontsize=12)
    plt.colorbar(im1, ax=ax1, label='Temperature (°C)')
    ax1.legend(loc='upper right', fontsize=8)
    
    # 2. Hot spot detection
    ax2 = plt.subplot(2, 3, 2)
    hot_spots = analyzer.detect_hot_spots(lst_data)
    ax2.imshow(hot_spots, cmap='Reds', alpha=0.7)
    ax2.imshow(lst_data, cmap='gray', alpha=0.3)
    ax2.set_title('Hot Spot Detection (>90th percentile)', fontweight='bold', fontsize=12)
    
    # 3. Zone statistics boxplot
    ax3 = plt.subplot(2, 3, 3)
    zone_stats = analyzer.zone_statistics(lst_data)
    zone_stats_sorted = zone_stats.sort_values('mean', ascending=False)
    
    positions = range(len(zone_stats_sorted))
    ax3.barh(positions, zone_stats_sorted['mean'], xerr=zone_stats_sorted['std'], 
            color='coral', alpha=0.7)
    ax3.set_yticks(positions)
    ax3.set_yticklabels([z.replace('_', ' ').title() for z in zone_stats_sorted['zone']])
    ax3.set_xlabel('Mean Temperature (°C)', fontweight='bold')
    ax3.set_title('Zone Temperature Comparison', fontweight='bold', fontsize=12)
    ax3.grid(axis='x', alpha=0.3)
    
    # 4. Thermal clustering
    ax4 = plt.subplot(2, 3, 4)
    cluster_map, centers = analyzer.thermal_clustering(lst_data, n_clusters=5)
    im4 = ax4.imshow(cluster_map, cmap='viridis')
    ax4.set_title('Thermal Zones (K-means Clustering)', fontweight='bold', fontsize=12)
    plt.colorbar(im4, ax=ax4, label='Cluster ID')
    
    # 5. Cooling corridors
    ax5 = plt.subplot(2, 3, 5)
    corridors = analyzer.identify_cooling_corridors(lst_data)
    ax5.imshow(lst_data, cmap='RdYlBu_r', vmin=10, vmax=50, alpha=0.5)
    ax5.imshow(corridors, cmap='Blues', alpha=0.6)
    ax5.set_title('Cooling Corridors', fontweight='bold', fontsize=12)
    
    # 6. Temperature gradient from urban core
    ax6 = plt.subplot(2, 3, 6)
    center = (850, 350)  # Approximate Portland urban center
    gradient_df = analyzer.temperature_gradient_analysis(lst_data, center, max_radius=400)
    
    ax6.plot(gradient_df['distance_km'], gradient_df['mean_temp'], 
            marker='o', linewidth=2, markersize=4)
    ax6.fill_between(gradient_df['distance_km'], 
                     gradient_df['mean_temp'] - gradient_df['std_temp'],
                     gradient_df['mean_temp'] + gradient_df['std_temp'],
                     alpha=0.3)
    ax6.set_xlabel('Distance from Urban Core (km)', fontweight='bold')
    ax6.set_ylabel('Mean Temperature (°C)', fontweight='bold')
    ax6.set_title('Urban-Rural Temperature Gradient', fontweight='bold', fontsize=12)
    ax6.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_comprehensive_analysis.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    return zone_stats