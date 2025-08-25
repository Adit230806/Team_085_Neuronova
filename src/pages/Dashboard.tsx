import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { 
  Shield, 
  TrendingUp, 
  AlertTriangle, 
  CheckCircle, 
  Upload, 
  Database,
  Activity,
  BarChart3,
  Clock,
  Users,
  Zap
} from 'lucide-react';
import { mockApi, PipelineStatus } from '@/services/mockApi';

const Dashboard = () => {
  const [pipelineStatus, setPipelineStatus] = useState<PipelineStatus | null>(null);
  const [stats, setStats] = useState({
    totalUploads: 47,
    anomaliesDetected: 156,
    securityScore: 98.2,
    activeUsers: 12
  });

  useEffect(() => {
    // Load pipeline status
    mockApi.getPipelineStatus().then(setPipelineStatus);
  }, []);

  const quickStats = [
    {
      title: 'Total Uploads',
      value: stats.totalUploads,
      icon: Upload,
      change: '+12%',
      changeType: 'positive' as const
    },
    {
      title: 'Anomalies Detected',
      value: stats.anomaliesDetected,
      icon: AlertTriangle,
      change: '+3%',
      changeType: 'negative' as const
    },
    {
      title: 'Security Score',
      value: `${stats.securityScore}%`,
      icon: Shield,
      change: '+0.8%',
      changeType: 'positive' as const
    },
    {
      title: 'Active Users',
      value: stats.activeUsers,
      icon: Users,
      change: '+2',
      changeType: 'positive' as const
    }
  ];

  const recentActivity = [
    { id: 1, action: 'Data Upload', user: 'admin', time: '2m ago', status: 'completed' },
    { id: 2, action: 'Pipeline Run', user: 'analyst1', time: '15m ago', status: 'completed' },
    { id: 3, action: 'Security Scan', user: 'system', time: '1h ago', status: 'completed' },
    { id: 4, action: 'Model Training', user: 'ml_engineer', time: '2h ago', status: 'running' }
  ];

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <div className="bg-gradient-to-r from-primary/10 via-accent/10 to-primary/10 border-b border-border">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-foreground">
                Welcome to SecureML Pipeline
              </h1>
              <p className="text-muted-foreground mt-2">
                Advanced machine learning security and anomaly detection platform
              </p>
            </div>
            <div className="flex items-center space-x-2">
              <Badge variant="outline" className="bg-success/10 text-success border-success">
                <CheckCircle className="w-3 h-3 mr-1" />
                System Healthy
              </Badge>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Quick Stats */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          {quickStats.map((stat, index) => {
            const Icon = stat.icon;
            return (
              <Card key={index} className="hover:shadow-md transition-shadow">
                <CardContent className="p-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-muted-foreground">
                        {stat.title}
                      </p>
                      <p className="text-2xl font-bold text-foreground">
                        {stat.value}
                      </p>
                      <p className={`text-xs mt-1 ${
                        stat.changeType === 'positive' ? 'text-success' : 'text-warning'
                      }`}>
                        {stat.change} from last week
                      </p>
                    </div>
                    <div className="p-3 bg-primary/10 rounded-full">
                      <Icon className="w-6 h-6 text-primary" />
                    </div>
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Pipeline Status */}
          <div className="lg:col-span-2">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Zap className="w-5 h-5 text-primary" />
                  <span>Pipeline Status</span>
                </CardTitle>
                <CardDescription>
                  Current status of the ML anomaly detection pipeline
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                {pipelineStatus && (
                  <>
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">Status</span>
                      <Badge 
                        variant={pipelineStatus.status === 'completed' ? 'default' : 'secondary'}
                        className={pipelineStatus.status === 'completed' ? 'bg-success text-success-foreground' : ''}
                      >
                        {pipelineStatus.status.toUpperCase()}
                      </Badge>
                    </div>
                    
                    <div className="space-y-2">
                      <div className="flex items-center justify-between text-sm">
                        <span>Progress</span>
                        <span>{pipelineStatus.progress}%</span>
                      </div>
                      <Progress value={pipelineStatus.progress} className="w-full" />
                    </div>

                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <span className="text-muted-foreground">Total Records</span>
                        <p className="font-semibold">{pipelineStatus.totalRecords.toLocaleString()}</p>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Anomalies Found</span>
                        <p className="font-semibold text-warning">{pipelineStatus.anomaliesDetected}</p>
                      </div>
                    </div>

                    <div className="flex items-center space-x-4">
                      <Link to="/results">
                        <Button className="flex items-center space-x-2">
                          <BarChart3 className="w-4 h-4" />
                          <span>View Results</span>
                        </Button>
                      </Link>
                      <Link to="/upload">
                        <Button variant="outline" className="flex items-center space-x-2">
                          <Upload className="w-4 h-4" />
                          <span>Upload Data</span>
                        </Button>
                      </Link>
                    </div>
                  </>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Recent Activity */}
          <div>
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Activity className="w-5 h-5 text-accent" />
                  <span>Recent Activity</span>
                </CardTitle>
                <CardDescription>
                  Latest system operations and user actions
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {recentActivity.map((activity) => (
                    <div key={activity.id} className="flex items-center space-x-3">
                      <div className={`w-2 h-2 rounded-full ${
                        activity.status === 'completed' ? 'bg-success' : 'bg-warning animate-pulse'
                      }`} />
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium text-foreground">
                          {activity.action}
                        </p>
                        <p className="text-xs text-muted-foreground">
                          by {activity.user} • {activity.time}
                        </p>
                      </div>
                      <Badge 
                        variant="outline" 
                        className={`text-xs ${
                          activity.status === 'completed' 
                            ? 'text-success border-success' 
                            : 'text-warning border-warning'
                        }`}
                      >
                        {activity.status}
                      </Badge>
                    </div>
                  ))}
                </div>
                <div className="mt-4 pt-4 border-t border-border">
                  <Link to="/logs">
                    <Button variant="ghost" size="sm" className="w-full">
                      View All Logs
                    </Button>
                  </Link>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Security Overview */}
        <div className="mt-8">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Shield className="w-5 h-5 text-success" />
                <span>Security Overview</span>
              </CardTitle>
              <CardDescription>
                Blockchain-style security logging and threat detection status
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="text-center">
                  <div className="w-12 h-12 bg-success/10 rounded-full flex items-center justify-center mx-auto mb-2">
                    <CheckCircle className="w-6 h-6 text-success" />
                  </div>
                  <h3 className="font-semibold text-foreground">Chain Integrity</h3>
                  <p className="text-sm text-success">100% Verified</p>
                </div>
                <div className="text-center">
                  <div className="w-12 h-12 bg-primary/10 rounded-full flex items-center justify-center mx-auto mb-2">
                    <Database className="w-6 h-6 text-primary" />
                  </div>
                  <h3 className="font-semibold text-foreground">Data Protection</h3>
                  <p className="text-sm text-primary">SHA-256 Encrypted</p>
                </div>
                <div className="text-center">
                  <div className="w-12 h-12 bg-accent/10 rounded-full flex items-center justify-center mx-auto mb-2">
                    <Clock className="w-6 h-6 text-accent" />
                  </div>
                  <h3 className="font-semibold text-foreground">Real-time Monitoring</h3>
                  <p className="text-sm text-accent">Active 24/7</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;