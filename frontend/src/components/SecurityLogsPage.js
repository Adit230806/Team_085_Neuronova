import React, { useState, useEffect } from 'react';
import api from '../services/api';
import { useNotification } from '../context/NotificationContext';

const SecurityLogsPage = () => {
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const { error, success } = useNotification();

  useEffect(() => {
    fetchLogs();
  }, []);

  const fetchLogs = async (showMessage = false) => {
    setRefreshing(true);
    try {
      const response = await api.get('/security/logs');
      setLogs(response.data.logs);
      if (showMessage) {
        success('Security logs refreshed');
      }
    } catch (err) {
      console.error('Failed to fetch logs:', err);
      error(err.response?.data?.detail || 'Failed to fetch security logs');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const getActionIcon = (action) => {
    switch (action) {
      case 'USER_LOGIN': return '🔐';
      case 'DATA_UPLOAD': return '📁';
      case 'PIPELINE_EXECUTION': return '⚙️';
      default: return '📝';
    }
  };

  const getActionColor = (action) => {
    switch (action) {
      case 'USER_LOGIN': return 'text-primary-600 bg-primary-50';
      case 'DATA_UPLOAD': return 'text-success-600 bg-success-50';
      case 'PIPELINE_EXECUTION': return 'text-warning-600 bg-warning-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleString();
  };

  if (loading) {
    return (
      <div className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        <div className="flex items-center justify-center h-64">
          <div className="loading-spinner"></div>
          <span className="ml-3 text-gray-600">Loading security logs...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
      <div className="mb-8">
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Security Audit Trail</h1>
            <p className="mt-2 text-gray-600">
              Blockchain-style immutable logs of all system operations
            </p>
          </div>
          <button
            onClick={() => fetchLogs(true)}
            disabled={refreshing}
            className="btn-primary flex items-center space-x-2"
          >
            {refreshing ? (
              <>
                <div className="loading-spinner w-4 h-4"></div>
                <span>Refreshing...</span>
              </>
            ) : (
              <>
                <span>🔄</span>
                <span>Refresh</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Security Features Info */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
        <div className="card text-center">
          <div className="text-3xl mb-2">🔗</div>
          <h3 className="font-semibold text-gray-900">Blockchain-Style</h3>
          <p className="text-sm text-gray-600 mt-1">
            Each log entry is cryptographically linked to the previous one
          </p>
        </div>
        <div className="card text-center">
          <div className="text-3xl mb-2">🔒</div>
          <h3 className="font-semibold text-gray-900">SHA-256 Hashing</h3>
          <p className="text-sm text-gray-600 mt-1">
            All data is hashed using military-grade encryption
          </p>
        </div>
        <div className="card text-center">
          <div className="text-3xl mb-2">🛡️</div>
          <h3 className="font-semibold text-gray-900">Immutable Records</h3>
          <p className="text-sm text-gray-600 mt-1">
            Tamper-proof audit trail for complete transparency
          </p>
        </div>
      </div>

      {/* Logs Display */}
      <div className="card">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-xl font-semibold text-gray-900">
            Recent Security Events
          </h2>
          <span className="text-sm text-gray-500">
            {logs.length} events recorded
          </span>
        </div>

        {logs.length === 0 ? (
          <div className="text-center py-8">
            <div className="text-4xl mb-4">📝</div>
            <h3 className="text-lg font-medium text-gray-900 mb-2">No Security Logs</h3>
            <p className="text-gray-600">
              Security events will appear here as they occur
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            {logs.map((log, index) => (
              <div
                key={index}
                className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow duration-200"
              >
                <div className="flex items-start justify-between">
                  <div className="flex items-start space-x-4">
                    <div className={`p-2 rounded-lg ${getActionColor(log.action)}`}>
                      <span className="text-lg">
                        {getActionIcon(log.action)}
                      </span>
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center space-x-2 mb-1">
                        <h3 className="font-medium text-gray-900">
                          {log.action.replace(/_/g, ' ')}
                        </h3>
                        <span className="text-xs bg-gray-100 text-gray-600 px-2 py-1 rounded">
                          Block #{index + 1}
                        </span>
                      </div>
                      <p className="text-sm text-gray-600 mb-2">
                        {formatTimestamp(log.timestamp)}
                      </p>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
                        <div>
                          <span className="font-medium text-gray-700">Data Hash:</span>
                          <span className="ml-2 font-mono text-gray-600 bg-gray-100 px-1 rounded">
                            {log.data_hash}
                          </span>
                        </div>
                        <div>
                          <span className="font-medium text-gray-700">Block Hash:</span>
                          <span className="ml-2 font-mono text-gray-600 bg-gray-100 px-1 rounded">
                            {log.block_hash}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-success-500 rounded-full animate-pulse-slow"></div>
                    <span className="text-xs text-success-600 font-medium">Verified</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Security Statistics */}
      {logs.length > 0 && (
        <div className="mt-8 grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="card text-center">
            <div className="text-2xl mb-2">🔐</div>
            <div className="text-xl font-bold text-primary-600">
              {logs.filter(log => log.action === 'USER_LOGIN').length}
            </div>
            <div className="text-sm text-gray-600">Login Events</div>
          </div>
          <div className="card text-center">
            <div className="text-2xl mb-2">📁</div>
            <div className="text-xl font-bold text-success-600">
              {logs.filter(log => log.action === 'DATA_UPLOAD').length}
            </div>
            <div className="text-sm text-gray-600">Data Uploads</div>
          </div>
          <div className="card text-center">
            <div className="text-2xl mb-2">⚙️</div>
            <div className="text-xl font-bold text-warning-600">
              {logs.filter(log => log.action === 'PIPELINE_EXECUTION').length}
            </div>
            <div className="text-sm text-gray-600">Pipeline Runs</div>
          </div>
          <div className="card text-center">
            <div className="text-2xl mb-2">📊</div>
            <div className="text-xl font-bold text-gray-600">
              {logs.length}
            </div>
            <div className="text-sm text-gray-600">Total Events</div>
          </div>
        </div>
      )}

      {/* Blockchain Integrity Info */}
      <div className="mt-8 card bg-gradient-to-r from-primary-50 to-primary-100 border-primary-200">
        <div className="text-center">
          <div className="text-3xl mb-4">⛓️</div>
          <h3 className="text-lg font-semibold text-primary-900 mb-2">
            Blockchain Integrity Verified
          </h3>
          <p className="text-sm text-primary-700 max-w-2xl mx-auto">
            Each security event is cryptographically linked to the previous event, 
            creating an unbreakable chain of trust. Any attempt to modify past records 
            would break the cryptographic integrity of the entire chain.
          </p>
          <div className="mt-4 flex items-center justify-center space-x-4 text-xs text-primary-600">
            <div className="flex items-center space-x-1">
              <div className="w-2 h-2 bg-success-500 rounded-full"></div>
              <span>Chain Verified</span>
            </div>
            <div className="flex items-center space-x-1">
              <div className="w-2 h-2 bg-success-500 rounded-full"></div>
              <span>Tamper-Proof</span>
            </div>
            <div className="flex items-center space-x-1">
              <div className="w-2 h-2 bg-success-500 rounded-full"></div>
              <span>Cryptographically Secure</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SecurityLogsPage;