import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import api from '../services/api';
import { useNotification } from '../context/NotificationContext';

const HomePage = () => {
  const [systemStatus, setSystemStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const { error } = useNotification();

  useEffect(() => {
    checkSystemHealth();
  }, []);

  const checkSystemHealth = async () => {
    try {
      const response = await api.get('/health');
      setSystemStatus({
        status: 'healthy',
        timestamp: response.data.timestamp
      });
    } catch (err) {
      setSystemStatus({
        status: 'error',
        message: 'Unable to connect to backend'
      });
      error('System health check failed');
    } finally {
      setLoading(false);
    }
  };

  const features = [
    {
      icon: '📁',
      title: 'Data Upload',
      description: 'Upload CSV or JSON files for analysis',
      link: '/upload',
      color: 'primary'
    },
    {
      icon: '⚙️',
      title: 'ML Pipeline',
      description: 'Run advanced anomaly detection algorithms',
      link: '/pipeline',
      color: 'success'
    },
    {
      icon: '📊',
      title: 'Results & Visualization',
      description: 'View detailed results with interactive charts',
      link: '/results',
      color: 'warning'
    },
    {
      icon: '🔒',
      title: 'Security Logs',
      description: 'Blockchain-style audit trail for all operations',
      link: '/security',
      color: 'danger'
    }
  ];

  const algorithms = [
    {
      name: 'Isolation Forest',
      description: 'Efficient anomaly detection for large datasets',
      use_case: 'Best for high-dimensional data'
    },
    {
      name: 'Local Outlier Factor',
      description: 'Density-based anomaly detection',
      use_case: 'Best for local anomaly patterns'
    }
  ];

  return (
    <div className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
      {/* Header */}
      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold text-gray-900 mb-4">
          Welcome to Secure ML Pipeline
        </h1>
        <p className="text-xl text-gray-600 max-w-3xl mx-auto">
          Advanced machine learning pipeline with enterprise-grade security, 
          real-time anomaly detection, and blockchain-style audit logging.
        </p>
      </div>

      {/* System Status */}
      <div className="mb-8">
        <div className="card">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className={`w-3 h-3 rounded-full ${
                loading ? 'bg-gray-400' : 
                systemStatus?.status === 'healthy' ? 'bg-success-500 animate-pulse-slow' : 'bg-danger-500'
              }`}></div>
              <h3 className="text-lg font-semibold text-gray-900">System Status</h3>
            </div>
            <div className="text-sm text-gray-500">
              {loading ? 'Checking...' : 
               systemStatus?.status === 'healthy' ? 'All systems operational' : 'System error'}
            </div>
          </div>
          {systemStatus?.timestamp && (
            <p className="text-xs text-gray-400 mt-2">
              Last checked: {new Date(systemStatus.timestamp).toLocaleString()}
            </p>
          )}
        </div>
      </div>

      {/* Features Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
        {features.map((feature, index) => (
          <Link
            key={index}
            to={feature.link}
            className="card hover:shadow-lg transition-shadow duration-200 group"
          >
            <div className="text-center">
              <div className="text-4xl mb-4 group-hover:animate-bounce-slow">
                {feature.icon}
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">
                {feature.title}
              </h3>
              <p className="text-sm text-gray-600">
                {feature.description}
              </p>
            </div>
          </Link>
        ))}
      </div>

      {/* ML Algorithms Info */}
      <div className="mb-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-6 text-center">
          Supported ML Algorithms
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {algorithms.map((algo, index) => (
            <div key={index} className="card">
              <h3 className="text-lg font-semibold text-gray-900 mb-2">
                {algo.name}
              </h3>
              <p className="text-gray-600 mb-3">
                {algo.description}
              </p>
              <div className="inline-block bg-primary-100 text-primary-800 text-xs px-2 py-1 rounded-full">
                {algo.use_case}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Security Features */}
      <div className="card bg-gradient-to-r from-primary-50 to-primary-100">
        <div className="text-center">
          <div className="text-3xl mb-4">🛡️</div>
          <h2 className="text-2xl font-bold text-gray-900 mb-4">
            Enterprise Security Features
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-left">
            <div>
              <h4 className="font-semibold text-gray-900 mb-2">JWT Authentication</h4>
              <p className="text-sm text-gray-600">
                Secure token-based authentication with automatic expiration
              </p>
            </div>
            <div>
              <h4 className="font-semibold text-gray-900 mb-2">SHA-256 Hashing</h4>
              <p className="text-sm text-gray-600">
                Cryptographic hashing for data integrity and security
              </p>
            </div>
            <div>
              <h4 className="font-semibold text-gray-900 mb-2">Blockchain Logging</h4>
              <p className="text-sm text-gray-600">
                Immutable audit trail with cryptographic proof of operations
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="mt-8 text-center">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h3>
        <div className="space-x-4">
          <Link to="/upload" className="btn-primary">
            Upload Data
          </Link>
          <Link to="/pipeline" className="btn-success">
            Run Pipeline
          </Link>
          <Link to="/results" className="btn-secondary">
            View Results
          </Link>
        </div>
      </div>
    </div>
  );
};

export default HomePage;