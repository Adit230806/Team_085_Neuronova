import React, { useEffect, useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Shield, Clock, Database, CheckCircle } from 'lucide-react';
import { mockApi, SecurityLog } from '@/services/mockApi';

const SecurityLogs = () => {
  const [logs, setLogs] = useState<SecurityLog[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    mockApi.getSecurityLogs().then(data => {
      setLogs(data.logs);
      setLoading(false);
    });
  }, []);

  if (loading) {
    return <div className="flex items-center justify-center min-h-screen">
      <div className="text-center">
        <div className="animate-pulse text-2xl mb-4">🔗</div>
        <p>Loading security logs...</p>
      </div>
    </div>;
  }

  return (
    <div className="min-h-screen bg-background p-4">
      <div className="max-w-7xl mx-auto space-y-8">
        <div className="text-center">
          <h1 className="text-3xl font-bold mb-2">Security Audit Trail</h1>
          <p className="text-muted-foreground">Blockchain-style immutable security logs</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <Card className="text-center">
            <CardContent className="p-6">
              <Shield className="w-8 h-8 text-success mx-auto mb-2" />
              <h3 className="font-semibold">Chain Integrity</h3>
              <p className="text-sm text-success">100% Verified</p>
            </CardContent>
          </Card>
          <Card className="text-center">
            <CardContent className="p-6">
              <Database className="w-8 h-8 text-primary mx-auto mb-2" />
              <h3 className="font-semibold">SHA-256 Hashing</h3>
              <p className="text-sm text-primary">Military Grade</p>
            </CardContent>
          </Card>
          <Card className="text-center">
            <CardContent className="p-6">
              <Clock className="w-8 h-8 text-accent mx-auto mb-2" />
              <h3 className="font-semibold">Real-time Monitoring</h3>
              <p className="text-sm text-accent">24/7 Active</p>
            </CardContent>
          </Card>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Security Events</CardTitle>
            <CardDescription>{logs.length} events recorded in the blockchain</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {logs.map((log, index) => (
                <div key={log.id} className="flex items-start space-x-4 p-4 border border-border rounded-lg hover:shadow-md transition-shadow">
                  <div className="w-2 h-2 bg-success rounded-full mt-2" />
                  <div className="flex-1">
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="font-medium">{log.action.replace(/_/g, ' ')}</h3>
                      <Badge variant="outline" className="text-xs">
                        Block #{index + 1}
                      </Badge>
                    </div>
                    <p className="text-sm text-muted-foreground mb-2">{log.details}</p>
                    <div className="flex items-center space-x-4 text-xs text-muted-foreground">
                      <span>by {log.user}</span>
                      <span>{new Date(log.timestamp).toLocaleString()}</span>
                      <Badge variant={log.status === 'verified' ? 'default' : 'secondary'} className="text-xs">
                        <CheckCircle className="w-3 h-3 mr-1" />
                        {log.status}
                      </Badge>
                    </div>
                    <div className="mt-2 text-xs font-mono text-muted-foreground">
                      Hash: {log.hash}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default SecurityLogs;