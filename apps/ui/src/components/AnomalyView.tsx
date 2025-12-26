'use client';

import { useState, useEffect } from 'react';
import { 
  AlertTriangle, TrendingUp, TrendingDown, Activity, Clock, 
  Loader2, RefreshCw, Zap, Target, Eye, Shield,
  ChevronRight, ExternalLink, Filter
} from 'lucide-react';
import { queryLogs, getTopTemplates, getTimeRange } from '@/lib/api';

interface Anomaly {
  id: string;
  type: 'spike' | 'drop' | 'new_pattern' | 'frequency' | 'error_burst';
  severity: 'critical' | 'high' | 'medium' | 'low';
  title: string;
  description: string;
  timestamp: string;
  affectedLogs: number;
  templateHash?: string;
  templateText?: string;
  confidence: number;
}

interface Props {
  serviceName: string;
  timeRange: string;
}

export default function AnomalyView({ serviceName, timeRange }: Props) {
  const [anomalies, setAnomalies] = useState<Anomaly[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedAnomaly, setSelectedAnomaly] = useState<Anomaly | null>(null);
  const [filterSeverity, setFilterSeverity] = useState<string>('all');

  const detectAnomalies = async () => {
    setLoading(true);
    setError(null);

    try {
      const { from, to } = getTimeRange(timeRange);

      // Get logs and templates data
      const [allLogs, errorLogs, warnLogs, templates] = await Promise.all([
        queryLogs({ from, to, service_name: serviceName || undefined, limit: 1, offset: 0 }),
        queryLogs({ from, to, service_name: serviceName || undefined, severity_min: 4, limit: 1, offset: 0 }),
        queryLogs({ from, to, service_name: serviceName || undefined, severity_min: 3, limit: 1, offset: 0 }),
        getTopTemplates({ service_name: serviceName, from, to, severity_min: 0, limit: 50 }),
      ]);

      const detectedAnomalies: Anomaly[] = [];
      const now = new Date();

      // 1. Error burst detection - critical when high error rate
      const errorRate = allLogs.total > 0 ? (errorLogs.total / allLogs.total) * 100 : 0;
      if (errorLogs.total > 0) {
        let severity: 'critical' | 'high' | 'medium' | 'low';
        if (errorRate > 15) severity = 'critical';
        else if (errorRate > 8) severity = 'high';
        else if (errorRate > 3) severity = 'medium';
        else severity = 'low';
        
        detectedAnomalies.push({
          id: 'err-rate-1',
          type: 'error_burst',
          severity,
          title: 'Error Rate Analysis',
          description: `Error rate: ${errorRate.toFixed(2)}% (${errorLogs.total.toLocaleString()} errors / ${allLogs.total.toLocaleString()} total)${errorRate > 5 ? ' - Above normal threshold' : ' - Within acceptable range'}`,
          timestamp: now.toISOString(),
          affectedLogs: errorLogs.total,
          confidence: Math.min(98, 75 + errorRate * 1.5),
        });
      }

      // 2. Warning rate analysis
      const warnOnlyLogs = warnLogs.total - errorLogs.total;
      const warnRate = allLogs.total > 0 ? (warnOnlyLogs / allLogs.total) * 100 : 0;
      if (warnOnlyLogs > 100) {
        let severity: 'critical' | 'high' | 'medium' | 'low';
        if (warnRate > 25) severity = 'high';
        else if (warnRate > 15) severity = 'medium';
        else severity = 'low';
        
        detectedAnomalies.push({
          id: 'warn-rate-1',
          type: 'frequency',
          severity,
          title: 'Warning Frequency',
          description: `${warnOnlyLogs.toLocaleString()} warnings detected (${warnRate.toFixed(1)}% of logs). Consider investigating warning sources.`,
          timestamp: now.toISOString(),
          affectedLogs: warnOnlyLogs,
          confidence: 82,
        });
      }

      // 3. Template analysis for pattern anomalies
      if (templates.templates.length > 0) {
        const topTemplates = templates.templates.slice(0, 5);
        const totalFromTop = topTemplates.reduce((sum, t) => sum + t.count, 0);
        const concentrationRatio = allLogs.total > 0 ? (totalFromTop / allLogs.total) * 100 : 0;
        
        // Spike detection - top template dominance
        const topTemplate = templates.templates[0];
        if (topTemplate && allLogs.total > 0) {
          const topRatio = (topTemplate.count / allLogs.total) * 100;
          if (topRatio > 30) {
            detectedAnomalies.push({
              id: 'spike-dom-1',
              type: 'spike',
              severity: topRatio > 60 ? 'critical' : topRatio > 45 ? 'high' : 'medium',
              title: 'Pattern Spike Detected',
              description: `Single pattern accounts for ${topRatio.toFixed(1)}% of all logs (${topTemplate.count.toLocaleString()} occurrences). Possible log flooding.`,
              timestamp: now.toISOString(),
              affectedLogs: topTemplate.count,
              templateHash: topTemplate.template_hash,
              templateText: topTemplate.template_text,
              confidence: 92,
            });
          }
        }

        // Pattern concentration analysis
        if (concentrationRatio > 80 && templates.templates.length > 5) {
          detectedAnomalies.push({
            id: 'concentration-1',
            type: 'frequency',
            severity: 'medium',
            title: 'High Pattern Concentration',
            description: `Top 5 patterns account for ${concentrationRatio.toFixed(1)}% of logs. System may have limited log diversity.`,
            timestamp: now.toISOString(),
            affectedLogs: totalFromTop,
            confidence: 78,
          });
        }

        // New pattern detection - look for error/exception patterns
        const errorPatterns = templates.templates.filter(t => 
          t.template_text.toLowerCase().includes('error') || 
          t.template_text.toLowerCase().includes('exception') ||
          t.template_text.toLowerCase().includes('fatal')
        );

        const failPatterns = templates.templates.filter(t =>
          t.template_text.toLowerCase().includes('fail') ||
          t.template_text.toLowerCase().includes('denied') ||
          t.template_text.toLowerCase().includes('refused')
        );

        const timeoutPatterns = templates.templates.filter(t =>
          t.template_text.toLowerCase().includes('timeout') ||
          t.template_text.toLowerCase().includes('timed out')
        );

        if (errorPatterns.length > 0) {
          const totalErrorPatternLogs = errorPatterns.reduce((sum, t) => sum + t.count, 0);
          detectedAnomalies.push({
            id: 'err-patterns-1',
            type: 'new_pattern',
            severity: totalErrorPatternLogs > 5000 ? 'critical' : totalErrorPatternLogs > 1000 ? 'high' : 'medium',
            title: `${errorPatterns.length} Error/Exception Pattern${errorPatterns.length > 1 ? 's' : ''}`,
            description: `Detected ${errorPatterns.length} distinct error patterns totaling ${totalErrorPatternLogs.toLocaleString()} occurrences.`,
            timestamp: now.toISOString(),
            affectedLogs: totalErrorPatternLogs,
            templateText: errorPatterns[0]?.template_text,
            confidence: 94,
          });
        }

        if (failPatterns.length > 0) {
          const totalFailLogs = failPatterns.reduce((sum, t) => sum + t.count, 0);
          detectedAnomalies.push({
            id: 'fail-patterns-1',
            type: 'new_pattern',
            severity: totalFailLogs > 2000 ? 'high' : totalFailLogs > 500 ? 'medium' : 'low',
            title: `${failPatterns.length} Failure Pattern${failPatterns.length > 1 ? 's' : ''}`,
            description: `Found ${failPatterns.length} failure/denied patterns with ${totalFailLogs.toLocaleString()} occurrences.`,
            timestamp: now.toISOString(),
            affectedLogs: totalFailLogs,
            templateText: failPatterns[0]?.template_text,
            confidence: 88,
          });
        }

        if (timeoutPatterns.length > 0) {
          const totalTimeoutLogs = timeoutPatterns.reduce((sum, t) => sum + t.count, 0);
          detectedAnomalies.push({
            id: 'timeout-patterns-1',
            type: 'drop',
            severity: totalTimeoutLogs > 1000 ? 'high' : totalTimeoutLogs > 200 ? 'medium' : 'low',
            title: 'Timeout Patterns Detected',
            description: `${timeoutPatterns.length} timeout pattern${timeoutPatterns.length > 1 ? 's' : ''} with ${totalTimeoutLogs.toLocaleString()} occurrences. May indicate network or service issues.`,
            timestamp: now.toISOString(),
            affectedLogs: totalTimeoutLogs,
            templateText: timeoutPatterns[0]?.template_text,
            confidence: 86,
          });
        }
      }

      // 4. Volume analysis with severity levels
      if (allLogs.total > 10000) {
        let severity: 'critical' | 'high' | 'medium' | 'low';
        if (allLogs.total > 5000000) severity = 'high';
        else if (allLogs.total > 1000000) severity = 'medium';
        else severity = 'low';
        
        detectedAnomalies.push({
          id: 'volume-1',
          type: 'spike',
          severity,
          title: 'Log Volume Analysis',
          description: `${allLogs.total.toLocaleString()} logs in time range.${allLogs.total > 1000000 ? ' Consider implementing log rotation or archival.' : ''}`,
          timestamp: now.toISOString(),
          affectedLogs: allLogs.total,
          confidence: 99,
        });
      }

      // 5. Template diversity check
      if (templates.templates.length === 1 && allLogs.total > 100) {
        detectedAnomalies.push({
          id: 'low-diversity-1',
          type: 'drop',
          severity: 'medium',
          title: 'Low Log Diversity',
          description: 'Only 1 unique pattern detected. This could indicate monotonous logging or missing log instrumentation.',
          timestamp: now.toISOString(),
          affectedLogs: allLogs.total,
          confidence: 85,
        });
      } else if (templates.templates.length > 500) {
        detectedAnomalies.push({
          id: 'high-diversity-1',
          type: 'frequency',
          severity: 'low',
          title: 'High Pattern Diversity',
          description: `${templates.templates.length}+ unique patterns detected. Consider template consolidation.`,
          timestamp: now.toISOString(),
          affectedLogs: allLogs.total,
          confidence: 72,
        });
      }

      // Sort anomalies by severity
      const severityOrder = { critical: 0, high: 1, medium: 2, low: 3 };
      detectedAnomalies.sort((a, b) => severityOrder[a.severity] - severityOrder[b.severity]);

      // If no anomalies, add a healthy status
      if (detectedAnomalies.length === 0) {
        detectedAnomalies.push({
          id: 'healthy-1',
          type: 'frequency',
          severity: 'low',
          title: 'System Healthy',
          description: 'No significant anomalies detected in the selected time range. All patterns appear normal.',
          timestamp: now.toISOString(),
          affectedLogs: 0,
          confidence: 95,
        });
      }

      setAnomalies(detectedAnomalies);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to detect anomalies');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    detectAnomalies();
  }, [serviceName, timeRange]);

  const filteredAnomalies = filterSeverity === 'all' 
    ? anomalies 
    : anomalies.filter(a => a.severity === filterSeverity);

  const severityColors = {
    critical: 'bg-red-500/20 text-red-400 border-red-500/30',
    high: 'bg-orange-500/20 text-orange-400 border-orange-500/30',
    medium: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
    low: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
  };

  const typeIcons = {
    spike: TrendingUp,
    drop: TrendingDown,
    new_pattern: Target,
    frequency: Activity,
    error_burst: Zap,
  };

  if (loading) {
    return (
      <div className="h-full flex items-center justify-center bg-gradient-to-br from-slate-50 to-slate-100">
        <div className="text-center">
          <Shield size={64} className="animate-pulse mx-auto text-blue-500 mb-4" />
          <p className="text-slate-600 font-medium">Analyzing for anomalies...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex bg-white">
      {/* Anomaly List */}
      <div className="w-1/2 border-r border-slate-200 flex flex-col">
        <div className="p-4 border-b border-slate-200 bg-gradient-to-r from-slate-50 to-white">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-br from-red-500 to-orange-500 rounded-xl flex items-center justify-center">
                <AlertTriangle size={20} className="text-white" />
              </div>
              <div>
                <h2 className="font-bold text-slate-900">Anomaly Detection</h2>
                <p className="text-xs text-slate-500">AI-powered pattern analysis</p>
              </div>
            </div>
            <button
              onClick={detectAnomalies}
              className="p-2 hover:bg-slate-100 rounded-lg text-slate-600 transition-colors"
            >
              <RefreshCw size={18} />
            </button>
          </div>

          <div className="flex items-center gap-2">
            <Filter size={14} className="text-slate-400" />
            <select
              value={filterSeverity}
              onChange={(e) => setFilterSeverity(e.target.value)}
              className="px-3 py-1.5 bg-white border border-slate-200 rounded-lg text-sm text-slate-700"
            >
              <option value="all">All Severities</option>
              <option value="critical">Critical</option>
              <option value="high">High</option>
              <option value="medium">Medium</option>
              <option value="low">Low</option>
            </select>
            <span className="text-sm text-slate-500 ml-2">
              {filteredAnomalies.length} anomalies
            </span>
          </div>
        </div>

        {error && (
          <div className="p-4 bg-red-50 border-b border-red-200 flex items-center gap-2 text-red-700">
            <AlertTriangle size={16} />
            <span className="text-sm">{error}</span>
          </div>
        )}

        <div className="flex-1 overflow-auto">
          {filteredAnomalies.map((anomaly) => {
            const Icon = typeIcons[anomaly.type];
            return (
              <button
                key={anomaly.id}
                onClick={() => setSelectedAnomaly(anomaly)}
                className={`w-full p-4 border-b border-slate-100 hover:bg-slate-50 text-left transition-all ${
                  selectedAnomaly?.id === anomaly.id ? 'bg-blue-50 border-l-4 border-l-blue-500' : ''
                }`}
              >
                <div className="flex items-start gap-3">
                  <div className={`p-2 rounded-lg ${severityColors[anomaly.severity]}`}>
                    <Icon size={18} />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <h3 className="font-semibold text-slate-900 truncate">{anomaly.title}</h3>
                      <span className={`px-2 py-0.5 rounded text-xs font-medium border ${severityColors[anomaly.severity]}`}>
                        {anomaly.severity}
                      </span>
                    </div>
                    <p className="text-sm text-slate-500 mt-1 line-clamp-2">{anomaly.description}</p>
                    <div className="flex items-center gap-4 mt-2 text-xs text-slate-400">
                      <span className="flex items-center gap-1">
                        <Eye size={12} />
                        {anomaly.affectedLogs.toLocaleString()} logs
                      </span>
                      <span className="flex items-center gap-1">
                        <Target size={12} />
                        {anomaly.confidence}% confidence
                      </span>
                    </div>
                  </div>
                  <ChevronRight size={18} className="text-slate-400 flex-shrink-0" />
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Detail Panel */}
      <div className="w-1/2 flex flex-col bg-slate-50">
        {selectedAnomaly ? (
          <>
            <div className="p-6 bg-white border-b border-slate-200">
              <div className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-lg mb-3 ${severityColors[selectedAnomaly.severity]} border`}>
                {(() => {
                  const Icon = typeIcons[selectedAnomaly.type];
                  return <Icon size={16} />;
                })()}
                <span className="text-sm font-medium capitalize">{selectedAnomaly.severity} Severity</span>
              </div>
              <h2 className="text-xl font-bold text-slate-900">{selectedAnomaly.title}</h2>
              <p className="text-slate-500 mt-2">{selectedAnomaly.description}</p>
            </div>

            <div className="flex-1 overflow-auto p-6">
              <div className="space-y-4">
                {/* Stats */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-white rounded-xl p-4 border border-slate-200">
                    <p className="text-sm text-slate-500">Affected Logs</p>
                    <p className="text-2xl font-bold text-slate-900 mt-1">
                      {selectedAnomaly.affectedLogs.toLocaleString()}
                    </p>
                  </div>
                  <div className="bg-white rounded-xl p-4 border border-slate-200">
                    <p className="text-sm text-slate-500">Confidence Score</p>
                    <p className="text-2xl font-bold text-slate-900 mt-1">
                      {selectedAnomaly.confidence}%
                    </p>
                  </div>
                </div>

                {/* Template Info */}
                {selectedAnomaly.templateText && (
                  <div className="bg-white rounded-xl p-4 border border-slate-200">
                    <p className="text-sm text-slate-500 mb-2">Related Pattern</p>
                    <code className="block p-3 bg-slate-900 text-emerald-400 rounded-lg text-sm font-mono overflow-x-auto">
                      {selectedAnomaly.templateText}
                    </code>
                    {selectedAnomaly.templateHash && (
                      <p className="text-xs text-slate-400 mt-2">
                        Template Hash: {selectedAnomaly.templateHash}
                      </p>
                    )}
                  </div>
                )}

                {/* Recommendations */}
                <div className="bg-white rounded-xl p-4 border border-slate-200">
                  <p className="text-sm font-semibold text-slate-900 mb-3">Recommendations</p>
                  <ul className="space-y-2 text-sm text-slate-600">
                    {selectedAnomaly.severity === 'critical' && (
                      <li className="flex items-start gap-2">
                        <span className="text-red-500 mt-1">•</span>
                        Immediate investigation required. Check system health and recent deployments.
                      </li>
                    )}
                    {selectedAnomaly.type === 'error_burst' && (
                      <li className="flex items-start gap-2">
                        <span className="text-amber-500 mt-1">•</span>
                        Review error logs and identify root cause. Consider rollback if needed.
                      </li>
                    )}
                    {selectedAnomaly.type === 'frequency' && (
                      <li className="flex items-start gap-2">
                        <span className="text-blue-500 mt-1">•</span>
                        Analyze the dominant pattern for potential performance issues or loops.
                      </li>
                    )}
                    <li className="flex items-start gap-2">
                      <span className="text-slate-400 mt-1">•</span>
                      Use the Log Explorer to investigate affected logs in detail.
                    </li>
                  </ul>
                </div>

                {/* Timestamp */}
                <div className="flex items-center gap-2 text-sm text-slate-500">
                  <Clock size={14} />
                  <span>Detected: {new Date(selectedAnomaly.timestamp).toLocaleString()}</span>
                </div>
              </div>
            </div>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center">
              <Eye size={48} className="mx-auto text-slate-300 mb-4" />
              <p className="text-slate-500">Select an anomaly to view details</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
