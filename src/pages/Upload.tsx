import React, { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  Upload as UploadIcon, 
  FileText, 
  CheckCircle, 
  AlertCircle,
  X,
  Download,
  Database,
  Shield
} from 'lucide-react';
import { mockApi } from '@/services/mockApi';
import { useToast } from '@/hooks/use-toast';

interface UploadResult {
  filename: string;
  size: number;
  records: number;
  columns: string[];
  uploadId: string;
}

const Upload = () => {
  const [dragActive, setDragActive] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadResult, setUploadResult] = useState<UploadResult | null>(null);
  const [error, setError] = useState('');

  const fileInputRef = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();
  const { toast } = useToast();

  const acceptedTypes = ['.csv', '.json', '.xlsx'];
  const maxFileSize = 50 * 1024 * 1024; // 50MB

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      handleFileSelection(files[0]);
    }
  };

  const handleFileSelection = (selectedFile: File) => {
    setError('');
    
    // Validate file type
    const fileExtension = '.' + selectedFile.name.split('.').pop()?.toLowerCase();
    if (!acceptedTypes.includes(fileExtension)) {
      setError(`Invalid file type. Please select a ${acceptedTypes.join(', ')} file.`);
      return;
    }

    // Validate file size
    if (selectedFile.size > maxFileSize) {
      setError('File size exceeds 50MB limit.');
      return;
    }

    setFile(selectedFile);
    setUploadResult(null);
  };

  const simulateUploadProgress = () => {
    setUploadProgress(0);
    const interval = setInterval(() => {
      setUploadProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          return 100;
        }
        return prev + Math.random() * 15;
      });
    }, 200);
    return interval;
  };

  const handleUpload = async () => {
    if (!file) return;

    setUploading(true);
    setError('');
    
    const progressInterval = simulateUploadProgress();

    try {
      const result = await mockApi.uploadFile(file);
      setUploadResult(result);
      toast({
        title: "Upload Successful",
        description: `${file.name} has been processed successfully.`,
      });
    } catch (err) {
      setError('Upload failed. Please try again.');
      toast({
        title: "Upload Failed",
        description: "There was an error uploading your file.",
        variant: "destructive",
      });
    } finally {
      clearInterval(progressInterval);
      setUploading(false);
      setUploadProgress(100);
    }
  };

  const clearFile = () => {
    setFile(null);
    setUploadResult(null);
    setError('');
    setUploadProgress(0);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="min-h-screen bg-background p-4">
      <div className="max-w-4xl mx-auto space-y-8">
        {/* Header */}
        <div className="text-center">
          <h1 className="text-3xl font-bold text-foreground mb-2">
            Data Upload Center
          </h1>
          <p className="text-muted-foreground">
            Upload your datasets for advanced anomaly detection and security analysis
          </p>
        </div>

        {/* Upload Area */}
        <Card className="shadow-lg">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <UploadIcon className="w-5 h-5 text-primary" />
              <span>Upload Dataset</span>
            </CardTitle>
            <CardDescription>
              Drag and drop your file or click to browse. Supported formats: CSV, JSON, XLSX
            </CardDescription>
          </CardHeader>
          <CardContent>
            {!file ? (
              <div
                className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
                  dragActive
                    ? 'border-primary bg-primary/5'
                    : 'border-muted-foreground/25 hover:border-primary/50 hover:bg-muted/50'
                }`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
              >
                <div className="space-y-4">
                  <div className="mx-auto w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center">
                    <UploadIcon className={`w-8 h-8 ${dragActive ? 'text-primary' : 'text-muted-foreground'}`} />
                  </div>
                  <div>
                    <h3 className="text-lg font-medium text-foreground">
                      {dragActive ? 'Drop your file here' : 'Upload your dataset'}
                    </h3>
                    <p className="text-muted-foreground">
                      Drag and drop or <span className="text-primary font-medium cursor-pointer">browse files</span>
                    </p>
                  </div>
                  <div className="flex flex-wrap gap-2 justify-center">
                    {acceptedTypes.map(type => (
                      <Badge key={type} variant="outline">
                        {type.toUpperCase()}
                      </Badge>
                    ))}
                  </div>
                </div>
                
                <input
                  ref={fileInputRef}
                  type="file"
                  accept={acceptedTypes.join(',')}
                  onChange={(e) => e.target.files && handleFileSelection(e.target.files[0])}
                  className="hidden"
                />
              </div>
            ) : (
              <div className="space-y-6">
                {/* File Info */}
                <div className="flex items-center justify-between p-4 bg-muted rounded-lg">
                  <div className="flex items-center space-x-3">
                    <FileText className="w-8 h-8 text-primary" />
                    <div>
                      <p className="font-medium text-foreground">{file.name}</p>
                      <p className="text-sm text-muted-foreground">
                        {formatFileSize(file.size)} • {file.type || 'Unknown type'}
                      </p>
                    </div>
                  </div>
                  <Button variant="ghost" size="sm" onClick={clearFile}>
                    <X className="w-4 h-4" />
                  </Button>
                </div>

                {/* Upload Progress */}
                {uploading && (
                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-sm">
                      <span>Uploading...</span>
                      <span>{Math.round(uploadProgress)}%</span>
                    </div>
                    <Progress value={uploadProgress} className="w-full" />
                  </div>
                )}

                {/* Error */}
                {error && (
                  <Alert variant="destructive">
                    <AlertCircle className="h-4 w-4" />
                    <AlertDescription>{error}</AlertDescription>
                  </Alert>
                )}

                {/* Upload Button */}
                {!uploadResult && (
                  <Button
                    onClick={handleUpload}
                    disabled={uploading}
                    className="w-full"
                    size="lg"
                  >
                    {uploading ? 'Processing...' : 'Upload & Analyze'}
                  </Button>
                )}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Upload Success */}
        {uploadResult && (
          <Card className="border-success bg-success/5">
            <CardHeader>
              <CardTitle className="flex items-center space-x-2 text-success">
                <CheckCircle className="w-5 h-5" />
                <span>Upload Successful!</span>
              </CardTitle>
              <CardDescription>
                Your dataset has been processed and is ready for analysis
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="text-center p-4 bg-card rounded-lg">
                  <Database className="w-8 h-8 text-primary mx-auto mb-2" />
                  <p className="text-2xl font-bold text-foreground">{uploadResult.records.toLocaleString()}</p>
                  <p className="text-sm text-muted-foreground">Records</p>
                </div>
                <div className="text-center p-4 bg-card rounded-lg">
                  <FileText className="w-8 h-8 text-accent mx-auto mb-2" />
                  <p className="text-2xl font-bold text-foreground">{uploadResult.columns.length}</p>
                  <p className="text-sm text-muted-foreground">Columns</p>
                </div>
                <div className="text-center p-4 bg-card rounded-lg">
                  <Shield className="w-8 h-8 text-success mx-auto mb-2" />
                  <p className="text-2xl font-bold text-foreground">100%</p>
                  <p className="text-sm text-muted-foreground">Secure</p>
                </div>
              </div>

              <div className="space-y-2">
                <h4 className="font-medium text-foreground">Column Names:</h4>
                <div className="flex flex-wrap gap-2">
                  {uploadResult.columns.map((column, index) => (
                    <Badge key={index} variant="secondary">
                      {column}
                    </Badge>
                  ))}
                </div>
              </div>

              <div className="flex space-x-4">
                <Button
                  onClick={() => navigate('/results')}
                  className="flex items-center space-x-2"
                >
                  <UploadIcon className="w-4 h-4" />
                  <span>Run Analysis</span>
                </Button>
                <Button variant="outline" onClick={clearFile}>
                  Upload Another File
                </Button>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Sample Data Info */}
        <Card>
          <CardHeader>
            <CardTitle>Sample Data Formats</CardTitle>
            <CardDescription>
              Examples of supported data formats for optimal processing
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h4 className="font-medium text-foreground mb-2">CSV Format</h4>
                <div className="bg-muted p-4 rounded-lg font-mono text-sm">
                  <div>id,feature1,feature2,category</div>
                  <div>1,2.5,1.8,normal</div>
                  <div>2,8.1,9.2,anomaly</div>
                  <div>3,1.9,2.1,normal</div>
                </div>
              </div>
              <div>
                <h4 className="font-medium text-foreground mb-2">JSON Format</h4>
                <div className="bg-muted p-4 rounded-lg font-mono text-sm">
                  <div>[</div>
                  <div>&nbsp;&nbsp;{"{"}"id": 1, "feature1": 2.5, "feature2": 1.8{"}"},</div>
                  <div>&nbsp;&nbsp;{"{"}"id": 2, "feature1": 8.1, "feature2": 9.2{"}"}</div>
                  <div>]</div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default Upload;