// Mock API service that provides sample data for demo purposes

export interface AnomalyResult {
  id: number;
  feature1: number;
  feature2: number;
  feature3: number;
  isAnomaly: boolean;
  anomalyScore: number;
  category: string;
}

export interface SecurityLog {
  id: string;
  timestamp: string;
  action: string;
  user: string;
  details: string;
  hash: string;
  blockHash: string;
  status: 'verified' | 'pending' | 'failed';
}

export interface PipelineStatus {
  status: 'idle' | 'running' | 'completed' | 'error';
  progress: number;
  message: string;
  totalRecords: number;
  anomaliesDetected: number;
  lastRun: string;
}

class MockApiService {
  private delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

  // Generate mock anomaly detection results
  generateMockResults(): AnomalyResult[] {
    const results: AnomalyResult[] = [];
    const categories = ['Financial', 'User Behavior', 'System Performance', 'Security'];
    
    for (let i = 1; i <= 100; i++) {
      const isAnomaly = Math.random() < 0.15; // 15% anomaly rate
      results.push({
        id: i,
        feature1: Math.random() * 100,
        feature2: Math.random() * 100 + (isAnomaly ? 50 : 0), // Anomalies have higher feature2
        feature3: Math.random() * 50,
        isAnomaly,
        anomalyScore: isAnomaly ? Math.random() * 0.4 + 0.6 : Math.random() * 0.4,
        category: categories[Math.floor(Math.random() * categories.length)]
      });
    }
    
    return results;
  }

  // Generate mock security logs
  generateMockLogs(): SecurityLog[] {
    const actions = ['USER_LOGIN', 'DATA_UPLOAD', 'PIPELINE_EXECUTION', 'MODEL_TRAINING', 'SECURITY_SCAN'];
    const users = ['admin', 'analyst1', 'datauser', 'ml_engineer'];
    const logs: SecurityLog[] = [];

    for (let i = 1; i <= 25; i++) {
      const timestamp = new Date(Date.now() - i * 3600000).toISOString(); // Each log 1 hour apart
      const action = actions[Math.floor(Math.random() * actions.length)];
      const user = users[Math.floor(Math.random() * users.length)];
      
      logs.push({
        id: `log_${i}`,
        timestamp,
        action,
        user,
        details: this.getActionDetails(action),
        hash: this.generateHash(),
        blockHash: this.generateHash(),
        status: Math.random() > 0.05 ? 'verified' : 'pending'
      });
    }

    return logs.reverse(); // Most recent first
  }

  private getActionDetails(action: string): string {
    const details = {
      'USER_LOGIN': 'Successful authentication with JWT token',
      'DATA_UPLOAD': 'CSV file uploaded and validated successfully',
      'PIPELINE_EXECUTION': 'Anomaly detection pipeline completed',
      'MODEL_TRAINING': 'ML model retrained with new data',
      'SECURITY_SCAN': 'Security vulnerability scan completed'
    };
    return details[action as keyof typeof details] || 'System operation completed';
  }

  private generateHash(): string {
    return Array.from({length: 16}, () => Math.floor(Math.random() * 16).toString(16)).join('');
  }

  // Mock API endpoints
  async login(username: string, password: string) {
    await this.delay(800);
    if (username === 'admin' && password === 'admin123') {
      return {
        success: true,
        token: 'mock-jwt-token-' + Date.now(),
        user: { username, role: 'Administrator' }
      };
    }
    return { success: false, error: 'Invalid credentials' };
  }

  async uploadFile(file: File) {
    await this.delay(2000);
    return {
      success: true,
      filename: file.name,
      size: file.size,
      records: Math.floor(Math.random() * 1000) + 500,
      columns: ['feature1', 'feature2', 'feature3', 'category'],
      uploadId: 'upload_' + Date.now()
    };
  }

  async runPipeline() {
    await this.delay(3000);
    return {
      success: true,
      jobId: 'job_' + Date.now(),
      status: 'completed',
      results: this.generateMockResults()
    };
  }

  async getPipelineStatus(): Promise<PipelineStatus> {
    await this.delay(500);
    return {
      status: 'completed',
      progress: 100,
      message: 'Analysis completed successfully',
      totalRecords: 1000,
      anomaliesDetected: 150,
      lastRun: new Date().toISOString()
    };
  }

  async getResults() {
    await this.delay(1000);
    return {
      success: true,
      data: this.generateMockResults(),
      summary: {
        totalRecords: 1000,
        anomaliesDetected: 150,
        confidenceScore: 0.92,
        processingTime: '2.3s'
      }
    };
  }

  async getSecurityLogs() {
    await this.delay(800);
    return {
      success: true,
      logs: this.generateMockLogs(),
      chainIntegrity: true,
      totalBlocks: 25
    };
  }
}

export const mockApi = new MockApiService();