"""
Keystroke Dynamics Feature Extraction Engine
Extracts behavioral biometric features from raw keystroke events
"""

import numpy as np
from typing import List, Dict, Any, Tuple
import statistics


def extract_features(keystroke_events: List[Dict]) -> Dict[str, float]:
    """
    Extract comprehensive feature vector from keystroke events.
    
    Events format: [{"key": "a", "type": "keydown"|"keyup", "timestamp": ms}, ...]
    
    Returns feature dictionary with 20+ behavioral metrics
    """
    if not keystroke_events or len(keystroke_events) < 4:
        return {}
    
    # Separate keydown and keyup events
    keydowns = {e['key']: e for e in keystroke_events if e['type'] == 'keydown'}
    keyups = {e['key']: e for e in keystroke_events if e['type'] == 'keyup'}
    
    # Compute hold times (dwell times) - time key is held down
    hold_times = []
    for key, down_event in keydowns.items():
        if key in keyups:
            hold = keyups[key]['timestamp'] - down_event['timestamp']
            if 0 < hold < 1000:  # sanity filter
                hold_times.append(hold)
    
    # Compute flight times (inter-key latency)
    flight_times = []
    sorted_downs = sorted(
        [e for e in keystroke_events if e['type'] == 'keydown'],
        key=lambda x: x['timestamp']
    )
    for i in range(1, len(sorted_downs)):
        flight = sorted_downs[i]['timestamp'] - sorted_downs[i-1]['timestamp']
        if 0 < flight < 2000:
            flight_times.append(flight)
    
    # Digraph latencies (pairs of consecutive keys)
    digraph_latencies = flight_times  # simplified
    
    # Compute total typing duration
    all_timestamps = [e['timestamp'] for e in keystroke_events]
    total_duration = max(all_timestamps) - min(all_timestamps) if all_timestamps else 0
    
    # Characters typed (keydown events only)
    char_count = len(sorted_downs)
    
    # Typing speed (chars per second)
    typing_speed = (char_count / (total_duration / 1000)) if total_duration > 0 else 0
    
    # Build feature vector
    features = {}
    
    # Hold time features
    if hold_times:
        features['hold_mean'] = float(np.mean(hold_times))
        features['hold_std'] = float(np.std(hold_times))
        features['hold_min'] = float(np.min(hold_times))
        features['hold_max'] = float(np.max(hold_times))
        features['hold_median'] = float(np.median(hold_times))
        features['hold_iqr'] = float(np.percentile(hold_times, 75) - np.percentile(hold_times, 25))
    else:
        for k in ['hold_mean','hold_std','hold_min','hold_max','hold_median','hold_iqr']:
            features[k] = 0.0
    
    # Flight time features
    if flight_times:
        features['flight_mean'] = float(np.mean(flight_times))
        features['flight_std'] = float(np.std(flight_times))
        features['flight_min'] = float(np.min(flight_times))
        features['flight_max'] = float(np.max(flight_times))
        features['flight_median'] = float(np.median(flight_times))
        features['flight_iqr'] = float(np.percentile(flight_times, 75) - np.percentile(flight_times, 25))
        features['flight_cv'] = float(np.std(flight_times) / np.mean(flight_times)) if np.mean(flight_times) > 0 else 0
    else:
        for k in ['flight_mean','flight_std','flight_min','flight_max','flight_median','flight_iqr','flight_cv']:
            features[k] = 0.0
    
    # Rhythm features
    features['typing_speed'] = float(typing_speed)
    features['total_duration'] = float(total_duration)
    features['char_count'] = float(char_count)
    
    # Rhythm consistency (coefficient of variation of inter-key intervals)
    if len(flight_times) > 1:
        cv = statistics.stdev(flight_times) / statistics.mean(flight_times) if statistics.mean(flight_times) > 0 else 0
        features['rhythm_consistency'] = float(1.0 / (1.0 + cv))  # 0-1 score, higher = more consistent
    else:
        features['rhythm_consistency'] = 0.5
    
    # Acceleration pattern (first half vs second half speed)
    if len(flight_times) >= 4:
        mid = len(flight_times) // 2
        first_half_speed = np.mean(flight_times[:mid])
        second_half_speed = np.mean(flight_times[mid:])
        features['acceleration_ratio'] = float(first_half_speed / second_half_speed) if second_half_speed > 0 else 1.0
    else:
        features['acceleration_ratio'] = 1.0
    
    # Pressure proxy (hold time variance as fatigue indicator)
    features['fatigue_index'] = features['hold_std'] / features['hold_mean'] if features['hold_mean'] > 0 else 0
    
    return features


def features_to_vector(features: Dict[str, float]) -> List[float]:
    """Convert feature dict to ordered numpy array for ML"""
    FEATURE_ORDER = [
        'hold_mean', 'hold_std', 'hold_min', 'hold_max', 'hold_median', 'hold_iqr',
        'flight_mean', 'flight_std', 'flight_min', 'flight_max', 'flight_median', 
        'flight_iqr', 'flight_cv',
        'typing_speed', 'total_duration', 'char_count',
        'rhythm_consistency', 'acceleration_ratio', 'fatigue_index'
    ]
    return [features.get(k, 0.0) for k in FEATURE_ORDER]


def compute_similarity_score(features1: Dict, features2: Dict) -> float:
    """Compute similarity between two feature sets (0-1)"""
    v1 = np.array(features_to_vector(features1))
    v2 = np.array(features_to_vector(features2))
    
    # Normalize vectors
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.5
    
    # Cosine similarity
    cosine_sim = np.dot(v1, v2) / (norm1 * norm2)
    return float((cosine_sim + 1) / 2)  # Normalize to 0-1
