import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import api from '../services/api';
import { useNotification } from '../context/NotificationContext';

const PipelineControlsPage = () => {
  const [algorithm, setAlgorithm] = useState('isolation_forest');
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState(null);
  const navigate = useNavigate();
  const { success, error, info } = useNotification();

  const algorithms = [
    {
      id: 'isolation_forest',
      name: 'Isolation Forest',
      description: 'Efficient anomaly detection algorithm that isolates anomalies by randomly selecting features and split values.',
      pros: ['Fast on large datasets', 'Works well with high-dimensional data', 'Low memory usage'],
      cons: ['May struggle with normal data clustering', 'Less interpretable results'],
      bestFor: 'Large datasets with mixed data types'
    },
    {
      id: 'lof',
      name: 'Local Outlier Factor (LOF)',
      description: 'Density-based algorithm that computes the local density deviation of a data point with respect to its neighbors.',
      pros: ['Good for local anomaly patterns', 'Handles varying densities well', 'More interpretable scores'],
      cons: ['Slower on large datasets', 'Sensitive to parameter selection'],
      bestFor: 'Datasets with varying local densities'
    }
  ];

  const runPipeline = async () => {
    setRunning(true);
    info('Starting anomaly detection pipeline...');

    try {
      const response = await api.post('/pipeline/run', {
        algorithm: algorithm
      });

      setResult(response.data);
      success(`Pipeline completed! Found ${response.data.anomalies_detected} anomalies with ${response.data.risk_level} risk level.`);
    } catch (err) {
      console.error('Pipeline failed:', err);
      error(err.response?.data?.detail || 'Pipeline execution failed');
    } finally {
      setRunning(false);
    }
  };

  const getRiskBadgeClass = (riskLevel) => {
    switch (riskLevel?.toLowerCase()) {
      case 'low': return 'badge-low';
      case 'medium': return 'badge-medium';
      case 'high': return 'badge-high';
      default: return 'badge-low';
    }
  };

  const selectedAlgo = algorithms.find(a => a.id === algorithm);

  return (
    <div className="max-w-6xl mx-auto py-6 sm:px-6 lg:px-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">ML Pipeline Controls</h1>
        <p className="mt-2 text-gray-600">
          Configure and execute machine learning anomaly detection algorithms
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Algorithm Selection */}
        <div className="lg:col-span-2">
          <div className="card">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">
              Select Algorithm
            </h2>
            
            <div className="space-y-4">
              {algorithms.map((algo) => (
                <div
                  key={algo.id}
                  className={`border rounded-lg p-4 cursor-pointer transition-all duration-200 ${
                    algorithm === algo.id
                      ? 'border-primary-500 bg-primary-50 ring-2 ring-primary-200'
                      : 'border-gray-200 hover:border-gray-300'
                  }`}
                  onClick={() => setAlgorithm(algo.id)}
                >
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex items-center space-x-3">
                      <input
                        type="radio"
                        checked={algorithm === algo.id}
                        onChange={() => setAlgorithm(algo.id)}
                        className="text-primary-600 focus:ring-primary-500"
                      />
                      <h3 className="text-lg font-medium text-gray-900">
                        {algo.name}
                      </h3>
                    </div>
                  </div>
                  
                  <p className="text-sm text-gray-600 mb-3 ml-7">
                    {algo.description}
                  </p>
                  
                  <div className="ml-7 grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
                    <div>
                      <h4 className="font-medium text-success-700 mb-1">Pros:</h4>
                      <ul className="list-disc list-inside text-success-600 space-y-1">
                        {algo.pros.map((pro, index) => (
                          <li key={index}>{pro}</li>
                        ))}
                      </ul>
                    </div>
                    <div>
                      <h4 className="font-medium text-warning-700 mb-1">Cons:</h4>
                      <ul className="list-disc list-inside text-warning-600 space-y-1">
                        {algo.cons.map((con, index) => (
                          <li key={index}>{con}</li>
                        ))}
                      </ul>
                    </div>
                  </div>
                  
                  <div className="ml-7 mt-3">
                    <span className="inline-block bg-gray-100 text-gray-800 text-xs px-2 py-1 rounded">
                      Best for: {algo.bestFor}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Pipeline Control Panel */}
        <div className="space-y-6">
          {/* Current Configuration */}
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
              Current Configuration
            </h3>
            <div className="space-y-3 text-sm">
              <div className="flex justify-between">
                <span className="font-medium text-gray-700">Algorithm:</span>
                <span className="text-gray-600">{selectedAlgo?.name}</span>
              </div>
              <div className="flex justify-between">
                <span className="font-medium text-gray-700">Status:</span>
                <span className={`font-medium ${running ? 'text-warning-600' : 'text-success-600'}`}>
                  {running ? 'Running...' : 'Ready'}
                </span>
              </div>
            </div>
          </div>

          {/* Pipeline Controls */}
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
              Pipeline Actions
            </h3>
            <div className="space-y-3">
              <button
                onClick={runPipeline}
                disabled={running}
                className="w-full btn-success flex items-center justify-center space-x-2"
              >
                {running ? (
                  <>
                    <div className="loading-spinner"></div>
                    <span>Running Pipeline...</span>
                  </>
                ) : (
                  <>
                    <span>▶️</span>
                    <span>Run Pipeline</span>
                  </>
                )}
              </button>
              
              <button
                onClick={() => navigate('/upload')}
                className="w-full btn-secondary"
              >
                📁 Upload New Data
              </button>
              
              {result && (
                <button
                  onClick={() => navigate('/results')}
                  className="w-full btn-primary"
                >
                  📊 View Results
                </button>
              )}
            </div>
          </div>

          {/* Last Run Results */}
          {result && (
            <div className="card bg-gray-50">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                Last Run Summary
              </h3>
              <div className="space-y-3 text-sm">
                <div className="flex justify-between">
                  <span className="font-medium text-gray-700">Algorithm:</span>
                  <span className="text-gray-600">{result.algorithm}</span>
                </div>
                <div className="flex justify-between">
                  <span className="font-medium text-gray-700">Total Records:</span>
                  <span className="text-gray-600">{result.total_records}</span>
                </div>
                <div className="flex justify-between">
                  <span className="font-medium text-gray-700">Anomalies:</span>
                  <span className="text-gray-600">{result.anomalies_detected}</span>
                </div>
                <div className="flex justify-between">
                  <span className="font-medium text-gray-700">Percentage:</span>
                  <span className="text-gray-600">{result.anomaly_percentage}%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="font-medium text-gray-700">Risk Level:</span>
                  <span className={getRiskBadgeClass(result.risk_level)}>
                    {result.risk_level}
                  </span>
                </div>
              </div>
            </div>
          )}

          {/* Pipeline Status */}
          <div className="card bg-primary-50 border-primary-200">
            <div className="flex items-center space-x-3">
              <div className="text-2xl">⚙️</div>
              <div>
                <h4 className="font-medium text-primary-900">Pipeline Status</h4>
                <p className="text-sm text-primary-700">
                  {running ? 'Processing your data...' : 'Ready to analyze data'}
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Progress Indicator */}
      {running && (
        <div className="fixed bottom-4 right-4 bg-white rounded-lg shadow-lg border p-4 max-w-sm">
          <div className="flex items-center space-x-3">
            <div className="loading-spinner"></div>
            <div>
              <p className="font-medium text-gray-900">Processing Data</p>
              <p className="text-sm text-gray-600">
                Running {selectedAlgo?.name}...
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default PipelineControlsPage;