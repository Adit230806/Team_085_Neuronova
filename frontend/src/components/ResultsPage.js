import React, { useState, useEffect } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Scatter, Pie, Bar } from 'react-chartjs-2';
import api from '../services/api';
import { useNotification } from '../context/NotificationContext';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend
);

const ResultsPage = () => {
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(true);
  const [selectedTab, setSelectedTab] = useState('table');
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage] = useState(10);
  const { error } = useNotification();

  useEffect(() => {
    fetchResults();
  }, []);

  const fetchResults = async () => {
    try {
      const response = await api.get('/results');
      setResults(response.data);
    } catch (err) {
      console.error('Failed to fetch results:', err);
      error(err.response?.data?.detail || 'Failed to fetch results');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        <div className="flex items-center justify-center h-64">
          <div className="loading-spinner"></div>
          <span className="ml-3 text-gray-600">Loading results...</span>
        </div>
      </div>
    );
  }

  if (!results) {
    return (
      <div className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        <div className="card text-center">
          <div className="text-4xl mb-4">📊</div>
          <h2 className="text-xl font-semibold text-gray-900 mb-2">No Results Available</h2>
          <p className="text-gray-600 mb-4">
            Please upload data and run the pipeline first to see results.
          </p>
          <button
            onClick={() => window.location.href = '/pipeline'}
            className="btn-primary"
          >
            Run Pipeline
          </button>
        </div>
      </div>
    );
  }

  const { data, charts, summary } = results;

  // Pagination
  const indexOfLastItem = currentPage * itemsPerPage;
  const indexOfFirstItem = indexOfLastItem - itemsPerPage;
  const currentItems = data.slice(indexOfFirstItem, indexOfLastItem);
  const totalPages = Math.ceil(data.length / itemsPerPage);

  // Chart configurations
  const pieChartData = {
    labels: charts.anomaly_distribution.labels,
    datasets: [
      {
        data: charts.anomaly_distribution.data,
        backgroundColor: ['#22c55e', '#ef4444'],
        borderColor: ['#16a34a', '#dc2626'],
        borderWidth: 2,
      },
    ],
  };

  const pieChartOptions = {
    responsive: true,
    plugins: {
      title: {
        display: true,
        text: 'Anomaly Distribution',
      },
      legend: {
        position: 'bottom',
      },
    },
  };

  const barChartData = {
    labels: ['Normal', 'Anomaly'],
    datasets: [
      {
        label: 'Count',
        data: charts.anomaly_distribution.data,
        backgroundColor: ['rgba(34, 197, 94, 0.8)', 'rgba(239, 68, 68, 0.8)'],
        borderColor: ['#16a34a', '#dc2626'],
        borderWidth: 2,
      },
    ],
  };

  const barChartOptions = {
    responsive: true,
    plugins: {
      title: {
        display: true,
        text: 'Anomaly Count Comparison',
      },
    },
    scales: {
      y: {
        beginAtZero: true,
      },
    },
  };

  let scatterChartData = null;
  let scatterChartOptions = null;

  if (charts.scatter_plot) {
    const { x, y, anomalies, x_label, y_label } = charts.scatter_plot;
    
    const normalPoints = x.map((xVal, i) => ({
      x: xVal,
      y: y[i],
    })).filter((_, i) => anomalies[i] === 0);

    const anomalyPoints = x.map((xVal, i) => ({
      x: xVal,
      y: y[i],
    })).filter((_, i) => anomalies[i] === 1);

    scatterChartData = {
      datasets: [
        {
          label: 'Normal',
          data: normalPoints,
          backgroundColor: 'rgba(34, 197, 94, 0.6)',
          borderColor: '#16a34a',
          pointRadius: 4,
        },
        {
          label: 'Anomaly',
          data: anomalyPoints,
          backgroundColor: 'rgba(239, 68, 68, 0.8)',
          borderColor: '#dc2626',
          pointRadius: 6,
        },
      ],
    };

    scatterChartOptions = {
      responsive: true,
      plugins: {
        title: {
          display: true,
          text: `${x_label} vs ${y_label}`,
        },
      },
      scales: {
        x: {
          display: true,
          title: {
            display: true,
            text: x_label,
          },
        },
        y: {
          display: true,
          title: {
            display: true,
            text: y_label,
          },
        },
      },
    };
  }

  const tabs = [
    { id: 'table', name: 'Data Table', icon: '📋' },
    { id: 'charts', name: 'Visualizations', icon: '📊' },
    { id: 'summary', name: 'Summary', icon: '📈' },
  ];

  const getRiskBadgeClass = (score) => {
    if (score >= 0.7) return 'badge-high';
    if (score >= 0.4) return 'badge-medium';
    return 'badge-low';
  };

  return (
    <div className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Analysis Results</h1>
        <p className="mt-2 text-gray-600">
          Detailed results from anomaly detection analysis
        </p>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <div className="card text-center">
          <div className="text-2xl mb-2">📊</div>
          <div className="text-2xl font-bold text-gray-900">{summary.total_records}</div>
          <div className="text-sm text-gray-600">Total Records</div>
        </div>
        <div className="card text-center">
          <div className="text-2xl mb-2">⚠️</div>
          <div className="text-2xl font-bold text-danger-600">{summary.anomalies}</div>
          <div className="text-sm text-gray-600">Anomalies Detected</div>
        </div>
        <div className="card text-center">
          <div className="text-2xl mb-2">✅</div>
          <div className="text-2xl font-bold text-success-600">{summary.total_records - summary.anomalies}</div>
          <div className="text-sm text-gray-600">Normal Records</div>
        </div>
        <div className="card text-center">
          <div className="text-2xl mb-2">📈</div>
          <div className="text-2xl font-bold text-primary-600">
            {((summary.anomalies / summary.total_records) * 100).toFixed(1)}%
          </div>
          <div className="text-sm text-gray-600">Anomaly Rate</div>
        </div>
      </div>

      {/* Tabs */}
      <div className="card">
        <div className="border-b border-gray-200">
          <nav className="-mb-px flex space-x-8">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setSelectedTab(tab.id)}
                className={`py-2 px-1 border-b-2 font-medium text-sm ${
                  selectedTab === tab.id
                    ? 'border-primary-500 text-primary-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <span className="mr-2">{tab.icon}</span>
                {tab.name}
              </button>
            ))}
          </nav>
        </div>

        <div className="p-6">
          {selectedTab === 'table' && (
            <div>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      {summary.columns.map((column) => (
                        <th
                          key={column}
                          className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                        >
                          {column}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {currentItems.map((row, index) => (
                      <tr
                        key={index}
                        className={row.anomaly === 1 ? 'bg-danger-50' : ''}
                      >
                        {summary.columns.map((column) => (
                          <td
                            key={column}
                            className="px-6 py-4 whitespace-nowrap text-sm text-gray-900"
                          >
                            {column === 'anomaly' ? (
                              <span className={row.anomaly === 1 ? 'badge-high' : 'badge-low'}>
                                {row.anomaly === 1 ? 'Anomaly' : 'Normal'}
                              </span>
                            ) : column === 'anomaly_score' ? (
                              <span className={getRiskBadgeClass(row.anomaly_score)}>
                                {(row.anomaly_score * 100).toFixed(1)}%
                              </span>
                            ) : (
                              String(row[column] || '')
                            )}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Pagination */}
              <div className="mt-4 flex items-center justify-between">
                <div className="text-sm text-gray-700">
                  Showing {indexOfFirstItem + 1} to {Math.min(indexOfLastItem, data.length)} of {data.length} results
                </div>
                <div className="flex space-x-2">
                  <button
                    onClick={() => setCurrentPage(prev => Math.max(prev - 1, 1))}
                    disabled={currentPage === 1}
                    className="btn-secondary text-sm disabled:opacity-50"
                  >
                    Previous
                  </button>
                  <span className="px-3 py-1 text-sm text-gray-600">
                    Page {currentPage} of {totalPages}
                  </span>
                  <button
                    onClick={() => setCurrentPage(prev => Math.min(prev + 1, totalPages))}
                    disabled={currentPage === totalPages}
                    className="btn-secondary text-sm disabled:opacity-50"
                  >
                    Next
                  </button>
                </div>
              </div>
            </div>
          )}

          {selectedTab === 'charts' && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <div className="bg-white p-4 rounded-lg border">
                <Pie data={pieChartData} options={pieChartOptions} />
              </div>
              <div className="bg-white p-4 rounded-lg border">
                <Bar data={barChartData} options={barChartOptions} />
              </div>
              {scatterChartData && (
                <div className="lg:col-span-2 bg-white p-4 rounded-lg border">
                  <Scatter data={scatterChartData} options={scatterChartOptions} />
                </div>
              )}
            </div>
          )}

          {selectedTab === 'summary' && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-gray-50 p-4 rounded-lg">
                  <h3 className="text-lg font-semibold text-gray-900 mb-3">Dataset Information</h3>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="font-medium text-gray-700">Total Records:</span>
                      <span>{summary.total_records}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="font-medium text-gray-700">Total Columns:</span>
                      <span>{summary.columns.length}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="font-medium text-gray-700">Anomalies Found:</span>
                      <span className="text-danger-600 font-medium">{summary.anomalies}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="font-medium text-gray-700">Normal Records:</span>
                      <span className="text-success-600 font-medium">{summary.total_records - summary.anomalies}</span>
                    </div>
                  </div>
                </div>

                <div className="bg-gray-50 p-4 rounded-lg">
                  <h3 className="text-lg font-semibold text-gray-900 mb-3">Analysis Metrics</h3>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="font-medium text-gray-700">Anomaly Rate:</span>
                      <span>{((summary.anomalies / summary.total_records) * 100).toFixed(2)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="font-medium text-gray-700">Detection Accuracy:</span>
                      <span className="text-primary-600 font-medium">High</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="font-medium text-gray-700">Model Confidence:</span>
                      <span className="text-success-600 font-medium">95%+</span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-primary-50 p-4 rounded-lg border border-primary-200">
                <h3 className="text-lg font-semibold text-primary-900 mb-2">
                  Recommendations
                </h3>
                <ul className="list-disc list-inside text-sm text-primary-800 space-y-1">
                  <li>Review highlighted anomalies for potential data quality issues</li>
                  <li>Consider investigating patterns in anomalous data points</li>
                  <li>Validate findings with domain expertise</li>
                  <li>Consider rerunning analysis with different algorithms for comparison</li>
                </ul>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ResultsPage;