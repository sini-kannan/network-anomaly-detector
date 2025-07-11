# network_anomaly_detector.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import json
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class NetworkAnomalyDetector:
    def __init__(self):
        self.data = None
        self.features = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.threat_levels = {
            'high': [],
            'medium': [],
            'low': []
        }
        
    def load_data(self, filepath):
        """Load network traffic data from CSV file"""
        try:
            self.data = pd.read_csv(filepath)
            print(f"Data loaded successfully. Shape: {self.data.shape}")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def generate_sample_data(self, n_samples=10000):
        """Generate realistic network traffic data for testing"""
        np.random.seed(42)
        
        # Create timestamps for the last 24 hours
        end_time = datetime.now()
        start_time = end_time - timedelta(days=1)
        timestamps = pd.date_range(start=start_time, end=end_time, periods=n_samples)
        
        # Normal traffic patterns (95% of data)
        normal_size = int(n_samples * 0.95)
        normal_data = {
            'timestamp': timestamps[:normal_size],
            'packet_size': np.random.normal(1500, 300, normal_size),
            'duration': np.random.exponential(0.5, normal_size),
            'bytes_sent': np.random.normal(5000, 1000, normal_size),
            'bytes_received': np.random.normal(3000, 800, normal_size),
            'protocol': np.random.choice(['TCP', 'UDP', 'HTTP', 'HTTPS'], normal_size, p=[0.4, 0.3, 0.2, 0.1]),
            'port': np.random.choice([80, 443, 21, 22, 25, 53], normal_size, p=[0.3, 0.3, 0.1, 0.1, 0.1, 0.1]),
            'source_ip': [f"192.168.1.{np.random.randint(1, 255)}" for _ in range(normal_size)],
            'dest_ip': [f"{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}" for _ in range(normal_size)]
        }
        
        # Anomalous traffic patterns (5% of data)
        anomaly_size = int(n_samples * 0.05)
        anomaly_data = {
            'timestamp': timestamps[normal_size:],
            'packet_size': np.random.normal(8000, 2000, anomaly_size),  # Unusually large packets
            'duration': np.random.exponential(10, anomaly_size),  # Long connections
            'bytes_sent': np.random.normal(50000, 15000, anomaly_size),  # High data transfer
            'bytes_received': np.random.normal(1000, 300, anomaly_size),  # Low response
            'protocol': np.random.choice(['TCP', 'UDP', 'Unknown'], anomaly_size, p=[0.5, 0.3, 0.2]),
            'port': np.random.choice([3389, 1433, 8080, 4444, 31337], anomaly_size),  # Suspicious ports
            'source_ip': [f"192.168.1.{np.random.randint(1, 255)}" for _ in range(anomaly_size)],
            'dest_ip': [f"{np.random.randint(1, 50)}.{np.random.randint(1, 50)}.{np.random.randint(1, 50)}.{np.random.randint(1, 255)}" for _ in range(anomaly_size)]
        }
        
        # Combine data
        all_data = {}
        for key in normal_data.keys():
            if key == 'timestamp':
                all_data[key] = np.concatenate([normal_data[key], anomaly_data[key]])
            else:
                all_data[key] = np.concatenate([normal_data[key], anomaly_data[key]])
        
        # Create labels (0 = normal, 1 = anomaly)
        labels = np.concatenate([np.zeros(normal_size), np.ones(anomaly_size)])
        
        # Create DataFrame
        self.data = pd.DataFrame(all_data)
        self.data['label'] = labels
        
        # Encode categorical variables
        self.data['protocol_encoded'] = pd.Categorical(self.data['protocol']).codes
        self.data['port_encoded'] = pd.Categorical(self.data['port']).codes
        
        # Add hour of day feature
        self.data['hour'] = pd.to_datetime(self.data['timestamp']).dt.hour
        
        print(f"Realistic sample data generated. Shape: {self.data.shape}")
        return self.data
    
    def preprocess_data(self):
        """Preprocess the data for anomaly detection"""
        if self.data is None:
            print("No data loaded. Please load data first.")
            return False
        
        # Select numerical features
        numerical_features = ['packet_size', 'duration', 'bytes_sent', 'bytes_received', 'hour']
        
        # Add encoded categorical features if they exist
        if 'protocol_encoded' in self.data.columns:
            numerical_features.extend(['protocol_encoded', 'port_encoded'])
        
        self.features = self.data[numerical_features].copy()
        
        # Handle missing values
        self.features = self.features.fillna(self.features.mean())
        
        # Scale features
        self.features_scaled = self.scaler.fit_transform(self.features)
        
        print(f"Data preprocessed. Features shape: {self.features.shape}")
        return True
    
    def train_isolation_forest(self, contamination=0.1):
        """Train Isolation Forest model"""
        model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        
        predictions = model.fit_predict(self.features_scaled)
        anomaly_scores = model.decision_function(self.features_scaled)
        predictions_binary = (predictions == -1).astype(int)
        
        self.models['isolation_forest'] = model
        self.results['isolation_forest'] = {
            'predictions': predictions_binary,
            'scores': anomaly_scores,
            'name': 'Isolation Forest'
        }
        
        return predictions_binary
    
    def train_local_outlier_factor(self, contamination=0.1):
        """Train Local Outlier Factor model"""
        model = LocalOutlierFactor(
            contamination=contamination,
            n_neighbors=20,
            novelty=True
        )
        
        model.fit(self.features_scaled)
        predictions = model.predict(self.features_scaled)
        anomaly_scores = model.decision_function(self.features_scaled)
        predictions_binary = (predictions == -1).astype(int)
        
        self.models['lof'] = model
        self.results['lof'] = {
            'predictions': predictions_binary,
            'scores': anomaly_scores,
            'name': 'Local Outlier Factor'
        }
        
        return predictions_binary
    
    def train_one_class_svm(self, nu=0.1):
        """Train One-Class SVM model"""
        model = OneClassSVM(
            nu=nu,
            kernel='rbf',
            gamma='scale'
        )
        
        predictions = model.fit_predict(self.features_scaled)
        anomaly_scores = model.decision_function(self.features_scaled)
        predictions_binary = (predictions == -1).astype(int)
        
        self.models['one_class_svm'] = model
        self.results['one_class_svm'] = {
            'predictions': predictions_binary,
            'scores': anomaly_scores,
            'name': 'One-Class SVM'
        }
        
        return predictions_binary
    
    def detect_anomalies(self):
        """Run all anomaly detection algorithms"""
        if self.features is None:
            print("Please preprocess data first.")
            return None
        
        print("Running anomaly detection algorithms...")
        
        # Train all models
        self.train_isolation_forest()
        self.train_local_outlier_factor()
        self.train_one_class_svm()
        
        # Ensemble prediction (majority voting)
        predictions = np.column_stack([
            self.results['isolation_forest']['predictions'],
            self.results['lof']['predictions'],
            self.results['one_class_svm']['predictions']
        ])
        
        ensemble_predictions = (predictions.sum(axis=1) >= 2).astype(int)
        self.results['ensemble'] = {
            'predictions': ensemble_predictions,
            'name': 'Ensemble Method'
        }
        
        # Classify threat levels
        self._classify_threats()
        
        print("Anomaly detection completed.")
        return ensemble_predictions
    
    def _classify_threats(self):
        """Classify anomalies by threat level"""
        if 'ensemble' not in self.results:
            return
        
        anomaly_indices = np.where(self.results['ensemble']['predictions'] == 1)[0]
        
        for idx in anomaly_indices:
            row = self.data.iloc[idx]
            
            # High threat indicators
            if (row['bytes_sent'] > 30000 or  # Large data transfer
                row['port'] in [3389, 1433, 4444, 31337] or  # Suspicious ports
                row['duration'] > 5):  # Long connections
                
                threat = {
                    'level': 'high',
                    'title': self._get_threat_title(row, 'high'),
                    'description': self._get_threat_description(row, 'high'),
                    'timestamp': row['timestamp'],
                    'source_ip': row['source_ip'],
                    'dest_ip': row['dest_ip'],
                    'port': row['port']
                }
                self.threat_levels['high'].append(threat)
            
            # Medium threat indicators
            elif (row['bytes_sent'] > 15000 or  # Moderate data transfer
                  row['packet_size'] > 3000 or  # Large packets
                  row['port'] in [8080, 21, 22]):  # Potentially risky ports
                
                threat = {
                    'level': 'medium',
                    'title': self._get_threat_title(row, 'medium'),
                    'description': self._get_threat_description(row, 'medium'),
                    'timestamp': row['timestamp'],
                    'source_ip': row['source_ip'],
                    'dest_ip': row['dest_ip'],
                    'port': row['port']
                }
                self.threat_levels['medium'].append(threat)
            
            # Low threat indicators
            else:
                threat = {
                    'level': 'low',
                    'title': self._get_threat_title(row, 'low'),
                    'description': self._get_threat_description(row, 'low'),
                    'timestamp': row['timestamp'],
                    'source_ip': row['source_ip'],
                    'dest_ip': row['dest_ip'],
                    'port': row['port']
                }
                self.threat_levels['low'].append(threat)
    
    def _get_threat_title(self, row, level):
        """Generate threat title based on anomaly characteristics"""
        if level == 'high':
            if row['bytes_sent'] > 30000:
                return "Potential Data Exfiltration"
            elif row['port'] in [3389, 1433]:
                return "Suspicious Remote Access"
            elif row['port'] in [4444, 31337]:
                return "Malware Communication"
            else:
                return "High-Risk Network Activity"
        
        elif level == 'medium':
            if row['packet_size'] > 3000:
                return "Unusual Packet Size"
            elif row['port'] in [8080, 21, 22]:
                return "Unusual Port Activity"
            else:
                return "Suspicious Network Pattern"
        
        else:
            return "Minor Network Anomaly"
    
    def _get_threat_description(self, row, level):
        """Generate threat description based on anomaly characteristics"""
        if level == 'high':
            if row['bytes_sent'] > 30000:
                return f"Large data transfer detected ({row['bytes_sent']/1000:.1f} KB to {row['dest_ip']})"
            elif row['port'] == 3389:
                return f"Remote Desktop connection from {row['source_ip']} to port {row['port']}"
            elif row['port'] == 1433:
                return f"Database connection attempt to {row['dest_ip']}:{row['port']}"
            else:
                return f"High-risk connection to {row['dest_ip']}:{row['port']}"
        
        elif level == 'medium':
            if row['packet_size'] > 3000:
                return f"Packet size {row['packet_size']} bytes exceeds normal range"
            elif row['port'] in [8080, 21, 22]:
                return f"Connection to {row['dest_ip']} on port {row['port']} flagged"
            else:
                return f"Unusual traffic pattern detected from {row['source_ip']}"
        
        else:
            return f"Minor anomaly in connection to {row['dest_ip']}:{row['port']}"
    
    def get_dashboard_data(self):
        """Get data formatted for the dashboard"""
        if not self.results:
            return None
        
        total_connections = len(self.data)
        normal_connections = int(np.sum(self.results['ensemble']['predictions'] == 0))
        suspicious_connections = int(np.sum(self.results['ensemble']['predictions'] == 1))
        threats_blocked = len(self.threat_levels['high'])
        
        # Calculate data transfer
        total_bytes = self.data['bytes_sent'].sum() + self.data['bytes_received'].sum()
        data_transfer_gb = total_bytes / (1024**3)
        
        # Hourly traffic data for chart
        hourly_data = self.data.groupby('hour').agg({
            'bytes_sent': 'sum',
            'bytes_received': 'sum'
        }).reset_index()
        
        # Threat distribution
        threat_distribution = {
            'safe': normal_connections,
            'low': len(self.threat_levels['low']),
            'medium': len(self.threat_levels['medium']),
            'high': len(self.threat_levels['high'])
        }
        
        # Recent threats for display
        recent_threats = []
        for level in ['high', 'medium', 'low']:
            for threat in self.threat_levels[level][:5]:  # Show last 5 of each type
                recent_threats.append(threat)
        
        # Sort by timestamp (most recent first)
        recent_threats.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return {
    'status': {
        'normal_connections': int(normal_connections),
        'suspicious_connections': int(suspicious_connections),
        'threats_blocked': int(threats_blocked),
        'data_transfer_gb': float(round(data_transfer_gb, 2))
    },
    'hourly_data': [
        {
            'hour': int(row['hour']),
            'bytes_sent': float(row['bytes_sent']),
            'bytes_received': float(row['bytes_received'])
        } for row in hourly_data.to_dict('records')
    ],
    'threat_distribution': {
        'safe': int(threat_distribution['safe']),
        'low': int(threat_distribution['low']),
        'medium': int(threat_distribution['medium']),
        'high': int(threat_distribution['high']),
    },
    'recent_threats': [
        {
            'level': t['level'],
            'title': t['title'],
            'description': t['description'],
            'timestamp': str(t['timestamp']),
            'source_ip': t['source_ip'],
            'dest_ip': t['dest_ip'],
            'port': int(t['port']),
        } for t in recent_threats[:10]
    ],
    'algorithm_performance': {
        name: {
            'anomalies': int(np.sum(results['predictions'])),
            'percentage': float(np.mean(results['predictions']) * 100)
        }
        for name, results in self.results.items()
    }
}



# web_app.py
from flask import Flask, render_template, jsonify, request, send_from_directory
import json
import os
from datetime import datetime

app = Flask(__name__)

# Global detector instance
detector = NetworkAnomalyDetector()
@app.route('/api/analyze')
def analyze_network():
    """API endpoint to run analysis and return dashboard data"""
    try:
        # Generate sample data
        detector.generate_sample_data()
        
        # Preprocess and analyze
        detector.preprocess_data()
        detector.detect_anomalies()
        
        # Get dashboard data
        dashboard_data = detector.get_dashboard_data()
        
        return jsonify(dashboard_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/deep-analysis')
def deep_analysis():
    """Run additional deep analysis"""
    # Add your deep analysis logic here
    return jsonify({'status': 'complete', 'findings': 'Additional patterns detected'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)