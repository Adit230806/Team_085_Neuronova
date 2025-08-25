import React, { useEffect, useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Pie } from 'react-chartjs-2';
import { Chart as ChartJS, ArcElement, Tooltip, Legend, CategoryScale, LinearScale, PointElement, LineElement } from 'chart.js';
import { AlertTriangle, CheckCircle, BarChart3, TrendingUp } from 'lucide-react';
import { mockApi, AnomalyResult } from '@/services/mockApi';

ChartJS.register(ArcElement, Tooltip, Legend, CategoryScale, LinearScale, PointElement, LineElement);

const Results = () => {
  const [results, setResults] = useState<AnomalyResult[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    mockApi.getResults().then(data => {
      setResults(data.data);
      setLoading(false);
    });
  }, []);

  if (loading) {
    return <div className="flex items-center justify-center min-h-screen">
      <div className="text-center">
        <div className="animate-pulse text-2xl mb-4">🔍</div>
        <p>Loading results...</p>
      </div>
    </div>;
  }

  const anomalies = results.filter(r => r.isAnomaly);
  const normal = results.filter(r => !r.isAnomaly);

  const chartData = {
    labels: ['Normal Data', 'Anomalies Detected'],
    datasets: [{
      data: [normal.length, anomalies.length],
      backgroundColor: ['hsl(var(--success))', 'hsl(var(--warning))'],
      borderWidth: 2,
    }]
  };

  return (
    <div className="min-h-screen bg-background p-4">
      <div className="max-w-7xl mx-auto space-y-8">
        <div className="text-center">
          <h1 className="text-3xl font-bold mb-2">Analysis Results</h1>
          <p className="text-muted-foreground">ML-powered anomaly detection findings</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <Card>
            <CardContent className="p-6 text-center">
              <CheckCircle className="w-8 h-8 text-success mx-auto mb-2" />
              <p className="text-2xl font-bold">{normal.length}</p>
              <p className="text-sm text-muted-foreground">Normal Records</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-6 text-center">
              <AlertTriangle className="w-8 h-8 text-warning mx-auto mb-2" />
              <p className="text-2xl font-bold">{anomalies.length}</p>
              <p className="text-sm text-muted-foreground">Anomalies Detected</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-6 text-center">
              <TrendingUp className="w-8 h-8 text-primary mx-auto mb-2" />
              <p className="text-2xl font-bold">94.2%</p>
              <p className="text-sm text-muted-foreground">Accuracy Score</p>
            </CardContent>
          </Card>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <Card>
            <CardHeader>
              <CardTitle>Distribution Overview</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="w-full h-64">
                <Pie data={chartData} options={{ maintainAspectRatio: false }} />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Top Anomalies</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {anomalies.slice(0, 5).map(anomaly => (
                  <div key={anomaly.id} className="flex items-center justify-between p-3 bg-muted rounded-lg">
                    <div>
                      <p className="font-medium">Record #{anomaly.id}</p>
                      <p className="text-sm text-muted-foreground">{anomaly.category}</p>
                    </div>
                    <Badge variant="destructive">
                      {(anomaly.anomalyScore * 100).toFixed(1)}%
                    </Badge>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default Results;