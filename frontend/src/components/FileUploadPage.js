import React, { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import api from '../services/api';
import { useNotification } from '../context/NotificationContext';

const FileUploadPage = () => {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);
  const navigate = useNavigate();
  const { success, error } = useNotification();

  const handleFile = (selectedFile) => {
    if (!selectedFile) return;

    // Validate file type
    const validTypes = ['text/csv', 'application/json'];
    const validExtensions = ['.csv', '.json'];
    
    const hasValidType = validTypes.includes(selectedFile.type);
    const hasValidExtension = validExtensions.some(ext => 
      selectedFile.name.toLowerCase().endsWith(ext)
    );

    if (!hasValidType && !hasValidExtension) {
      error('Please select a CSV or JSON file');
      return;
    }

    // Validate file size (max 10MB)
    if (selectedFile.size > 10 * 1024 * 1024) {
      error('File size must be less than 10MB');
      return;
    }

    setFile(selectedFile);
    setUploadResult(null);
  };

  const handleFileSelect = (e) => {
    const selectedFile = e.target.files?.[0];
    handleFile(selectedFile);
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    const droppedFile = e.dataTransfer.files?.[0];
    handleFile(droppedFile);
  };

  const uploadFile = async () => {
    if (!file) {
      error('Please select a file first');
      return;
    }

    setUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await api.post('/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setUploadResult(response.data);
      success('File uploaded successfully!');
    } catch (err) {
      console.error('Upload failed:', err);
      error(err.response?.data?.detail || 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="max-w-4xl mx-auto py-6 sm:px-6 lg:px-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Data Upload</h1>
        <p className="mt-2 text-gray-600">
          Upload your CSV or JSON data files for anomaly detection analysis
        </p>
      </div>

      {/* Upload Area */}
      <div className="card mb-8">
        <div
          className={`relative border-2 border-dashed rounded-lg p-6 transition-colors duration-200 ${
            dragActive
              ? 'border-primary-400 bg-primary-50'
              : 'border-gray-300 hover:border-gray-400'
          }`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          <div className="text-center">
            <div className="text-4xl mb-4">
              {dragActive ? '⬇️' : '📁'}
            </div>
            
            {!file ? (
              <>
                <h3 className="text-lg font-medium text-gray-900 mb-2">
                  {dragActive ? 'Drop your file here' : 'Upload your data file'}
                </h3>
                <p className="text-sm text-gray-500 mb-4">
                  Drag and drop your file here, or click to browse
                </p>
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="btn-primary"
                >
                  Select File
                </button>
              </>
            ) : (
              <div className="space-y-4">
                <div className="flex items-center justify-center space-x-2">
                  <span className="text-2xl">📄</span>
                  <div className="text-left">
                    <p className="font-medium text-gray-900">{file.name}</p>
                    <p className="text-sm text-gray-500">
                      {formatFileSize(file.size)} • {file.type || 'Unknown type'}
                    </p>
                  </div>
                </div>
                <div className="flex space-x-2">
                  <button onClick={uploadFile} disabled={uploading} className="btn-success">
                    {uploading ? (
                      <div className="flex items-center space-x-2">
                        <div className="loading-spinner"></div>
                        <span>Uploading...</span>
                      </div>
                    ) : (
                      'Upload File'
                    )}
                  </button>
                  <button
                    onClick={() => {
                      setFile(null);
                      setUploadResult(null);
                    }}
                    className="btn-secondary"
                  >
                    Clear
                  </button>
                </div>
              </div>
            )}
          </div>

          <input
            ref={fileInputRef}
            type="file"
            accept=".csv,.json"
            onChange={handleFileSelect}
            className="hidden"
          />
        </div>

        <div className="text-xs text-gray-500 space-y-1">
          <p>• Supported formats: CSV, JSON</p>
          <p>• Maximum file size: 10MB</p>
          <p>• Data will be processed securely and logged in the blockchain audit trail</p>
        </div>
      </div>

      {/* Upload Result */}
      {uploadResult && (
        <div className="card bg-success-50 border-success-200">
          <div className="flex items-start space-x-3">
            <div className="text-2xl">✅</div>
            <div className="flex-1">
              <h3 className="text-lg font-semibold text-success-800 mb-2">
                Upload Successful!
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                <div>
                  <span className="font-medium text-success-700">Rows:</span>
                  <span className="ml-2 text-success-600">{uploadResult.rows}</span>
                </div>
                <div>
                  <span className="font-medium text-success-700">Columns:</span>
                  <span className="ml-2 text-success-600">{uploadResult.columns}</span>
                </div>
                <div>
                  <span className="font-medium text-success-700">Status:</span>
                  <span className="ml-2 text-success-600">Ready for analysis</span>
                </div>
              </div>
              
              {uploadResult.column_names && (
                <div className="mt-4">
                  <p className="font-medium text-success-700 mb-2">Column Names:</p>
                  <div className="flex flex-wrap gap-2">
                    {uploadResult.column_names.map((col, index) => (
                      <span
                        key={index}
                        className="inline-block bg-success-100 text-success-800 text-xs px-2 py-1 rounded"
                      >
                        {col}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              <div className="mt-4 flex space-x-2">
                <button
                  onClick={() => navigate('/pipeline')}
                  className="btn-success text-sm"
                >
                  Run Analysis →
                </button>
                <button
                  onClick={() => setUploadResult(null)}
                  className="btn-secondary text-sm"
                >
                  Upload Another File
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Sample Data Info */}
      <div className="card bg-gray-50">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Sample Data Format
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm">
          <div>
            <h4 className="font-medium text-gray-700 mb-2">CSV Format Example:</h4>
            <div className="bg-white p-3 rounded border font-mono text-xs">
              <div>id,feature1,feature2,category</div>
              <div>1,2.1,3.4,A</div>
              <div>2,1.8,2.9,A</div>
              <div>3,8.1,9.4,B</div>
            </div>
          </div>
          <div>
            <h4 className="font-medium text-gray-700 mb-2">JSON Format Example:</h4>
            <div className="bg-white p-3 rounded border font-mono text-xs">
              <div>[</div>
              <div>&nbsp;&nbsp;{"{"}"id": 1, "feature1": 2.1, "feature2": 3.4{"}"},</div>
              <div>&nbsp;&nbsp;{"{"}"id": 2, "feature1": 1.8, "feature2": 2.9{"}"}</div>
              <div>]</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default FileUploadPage;