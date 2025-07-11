from flask import Flask, render_template, jsonify
from network_anomaly_detector import NetworkAnomalyDetector

app = Flask(__name__)
detector = NetworkAnomalyDetector()

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/analyze')
def analyze():
    detector.generate_sample_data()
    detector.preprocess_data()
    detector.detect_anomalies()
    data = detector.get_dashboard_data()
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
