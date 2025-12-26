'use client';

import { useState, useEffect } from 'react';
import { 
  BarChart2, TrendingUp, AlertTriangle, Clock, 
  Activity, Server, FileText, Loader2, RefreshCw,
  ArrowUp, ArrowDown, Minus
} from 'lucide-react';
import { queryLogs, getTopTemplates, getTimeRange } from '@/lib/api';

interface Props {
  serviceName: string;
  timeRange: string;
}

interface Stats {
  totalLogs: number;
  errorCount: number;
  warnCount: number;
  infoCount: number;
  debugCount: number;
  uniqueTemplates: number;
  topErrorTemplates: Array<{
    template_hash: string;  // String for 64-bit ints
    template_text: string;
    count: number;
  }>;
  logsPerHour: Array<{ hour: string; count: number }>;
}

export default function StatsView({ serviceName, timeRange }: Props) {
  const [stats, setStats] = useState<Stats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadStats = async () => {
    setLoading(true);
    setError(null);

    try {
      const { from, to } = getTimeRange(timeRange);

      // Get logs with different severity levels in parallel
      const [allLogs, errorTemplates] = await Promise.all([
        queryLogs({ from, to, service_name: serviceName || undefined, limit: 1, offset: 0 }),
        getTopTemplates({
          service_name: serviceName,
          from,
          to,
          severity_min: 4, // ERROR+
          limit: 10,
        }),
      ]);

      // Get severity breakdown
      const [errorLogs, warnLogs, infoLogs] = await Promise.all([
        queryLogs({ from, to, service_name: serviceName || undefined, severity_min: 4, limit: 1, offset: 0 }),
        queryLogs({ from, to, service_name: serviceName || undefined, severity_min: 3, limit: 1, offset: 0 }),
        queryLogs({ from, to, service_name: serviceName || undefined, severity_min: 2, limit: 1, offset: 0 }),
      ]);

      setStats({
        totalLogs: allLogs.total,
        errorCount: errorLogs.total,
        warnCount: warnLogs.total - errorLogs.total,
        infoCount: infoLogs.total - warnLogs.total,
        debugCount: allLogs.total - infoLogs.total,
        uniqueTemplates: errorTemplates.total,
        topErrorTemplates: errorTemplates.templates.slice(0, 5),
        logsPerHour: [], // Would need separate API
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load statistics');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadStats();
  }, [serviceName, timeRange]);

  if (loading) {
    return (
      <div className="h-full flex items-center justify-center bg-white">
        <div className="text-center">
          <Loader2 size={48} className="animate-spin mx-auto text-blue-500 mb-4" />
          <p className="text-slate-600 font-medium">Loading statistics...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="h-full flex items-center justify-center bg-white">
        <div className="text-center">
          <AlertTriangle size={48} className="mx-auto text-red-500 mb-4" />
          <p className="text-slate-800 font-medium">{error}</p>
          <button
            onClick={loadStats}
            className="mt-4 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 flex items-center gap-2 mx-auto"
          >
            <RefreshCw size={16} /> Retry
          </button>
        </div>
      </div>
    );
  }

  if (!stats) return null;

  const severityData = [
    { label: 'ERROR', count: stats.errorCount, color: 'bg-red-500', textColor: 'text-red-600' },
    { label: 'WARN', count: stats.warnCount, color: 'bg-amber-500', textColor: 'text-amber-600' },
    { label: 'INFO', count: stats.infoCount, color: 'bg-blue-500', textColor: 'text-blue-600' },
    { label: 'DEBUG', count: stats.debugCount, color: 'bg-slate-400', textColor: 'text-slate-600' },
  ];

  const maxCount = Math.max(...severityData.map(s => s.count));

  return (
    <div className="h-full overflow-auto bg-slate-50 p-6">
      <div className="max-w-6xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-slate-900">Log Statistics</h1>
            <p className="text-slate-500 mt-1">
              {serviceName ? `Service: ${serviceName}` : 'All Services'} • {timeRange}
            </p>
          </div>
          <button
            onClick={loadStats}
            className="px-4 py-2 bg-white border border-slate-200 rounded-lg hover:bg-slate-50 flex items-center gap-2 text-slate-700 shadow-sm"
          >
            <RefreshCw size={16} /> Refresh
          </button>
        </div>

        {/* Summary Cards */}
        <div className="grid grid-cols-4 gap-4">
          <div className="bg-white rounded-xl p-5 border border-slate-200 shadow-sm">
            <div className="flex items-center justify-between">
              <Activity size={24} className="text-blue-500" />
              <span className="text-xs text-slate-400 uppercase">Total</span>
            </div>
            <p className="text-3xl font-bold text-slate-900 mt-3">
              {stats.totalLogs.toLocaleString()}
            </p>
            <p className="text-sm text-slate-500 mt-1">Log Events</p>
          </div>

          <div className="bg-white rounded-xl p-5 border border-red-200 shadow-sm">
            <div className="flex items-center justify-between">
              <AlertTriangle size={24} className="text-red-500" />
              <span className="text-xs text-red-400 uppercase">Critical</span>
            </div>
            <p className="text-3xl font-bold text-red-600 mt-3">
              {stats.errorCount.toLocaleString()}
            </p>
            <p className="text-sm text-slate-500 mt-1">Errors</p>
          </div>

          <div className="bg-white rounded-xl p-5 border border-amber-200 shadow-sm">
            <div className="flex items-center justify-between">
              <AlertTriangle size={24} className="text-amber-500" />
              <span className="text-xs text-amber-400 uppercase">Warning</span>
            </div>
            <p className="text-3xl font-bold text-amber-600 mt-3">
              {stats.warnCount.toLocaleString()}
            </p>
            <p className="text-sm text-slate-500 mt-1">Warnings</p>
          </div>

          <div className="bg-white rounded-xl p-5 border border-slate-200 shadow-sm">
            <div className="flex items-center justify-between">
              <FileText size={24} className="text-emerald-500" />
              <span className="text-xs text-slate-400 uppercase">Patterns</span>
            </div>
            <p className="text-3xl font-bold text-slate-900 mt-3">
              {stats.uniqueTemplates.toLocaleString()}
            </p>
            <p className="text-sm text-slate-500 mt-1">Unique Templates</p>
          </div>
        </div>

        {/* Two columns */}
        <div className="grid grid-cols-2 gap-6">
          {/* Severity Distribution */}
          <div className="bg-white rounded-xl p-6 border border-slate-200 shadow-sm">
            <h2 className="text-lg font-bold text-slate-900 mb-4 flex items-center gap-2">
              <BarChart2 size={20} className="text-blue-500" />
              Severity Distribution
            </h2>
            <div className="space-y-4">
              {severityData.map((sev) => (
                <div key={sev.label}>
                  <div className="flex items-center justify-between mb-1">
                    <span className={`text-sm font-semibold ${sev.textColor}`}>{sev.label}</span>
                    <span className="text-sm text-slate-600 font-medium">
                      {sev.count.toLocaleString()}
                    </span>
                  </div>
                  <div className="h-3 bg-slate-100 rounded-full overflow-hidden">
                    <div
                      className={`h-full ${sev.color} rounded-full transition-all duration-500`}
                      style={{ width: `${maxCount > 0 ? (sev.count / maxCount) * 100 : 0}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Top Error Patterns */}
          <div className="bg-white rounded-xl p-6 border border-slate-200 shadow-sm">
            <h2 className="text-lg font-bold text-slate-900 mb-4 flex items-center gap-2">
              <TrendingUp size={20} className="text-red-500" />
              Top Error Patterns
            </h2>
            {stats.topErrorTemplates.length > 0 ? (
              <div className="space-y-3">
                {stats.topErrorTemplates.map((template, idx) => (
                  <div
                    key={template.template_hash}
                    className="p-3 bg-slate-50 rounded-lg border border-slate-100"
                  >
                    <div className="flex items-start gap-3">
                      <span className="flex-shrink-0 w-6 h-6 bg-red-100 text-red-600 rounded-full flex items-center justify-center text-xs font-bold">
                        {idx + 1}
                      </span>
                      <div className="flex-1 min-w-0">
                        <p className="text-sm text-slate-800 font-mono truncate">
                          {template.template_text}
                        </p>
                        <p className="text-xs text-red-600 font-semibold mt-1">
                          {template.count.toLocaleString()} occurrences
                        </p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8 text-slate-500">
                <AlertTriangle size={32} className="mx-auto text-slate-300 mb-2" />
                No errors found in this time range
              </div>
            )}
          </div>
        </div>

        {/* Error Rate */}
        <div className="bg-white rounded-xl p-6 border border-slate-200 shadow-sm">
          <h2 className="text-lg font-bold text-slate-900 mb-4 flex items-center gap-2">
            <Activity size={20} className="text-emerald-500" />
            Health Overview
          </h2>
          <div className="grid grid-cols-3 gap-6">
            <div className="text-center p-4 bg-slate-50 rounded-xl">
              <p className="text-4xl font-bold text-slate-900">
                {stats.totalLogs > 0 
                  ? ((stats.errorCount / stats.totalLogs) * 100).toFixed(2) 
                  : '0.00'}%
              </p>
              <p className="text-sm text-slate-500 mt-1">Error Rate</p>
            </div>
            <div className="text-center p-4 bg-slate-50 rounded-xl">
              <p className="text-4xl font-bold text-slate-900">
                {stats.totalLogs > 0 
                  ? (((stats.errorCount + stats.warnCount) / stats.totalLogs) * 100).toFixed(2) 
                  : '0.00'}%
              </p>
              <p className="text-sm text-slate-500 mt-1">Error + Warn Rate</p>
            </div>
            <div className="text-center p-4 bg-slate-50 rounded-xl">
              <p className="text-4xl font-bold text-emerald-600">
                {stats.totalLogs > 0 
                  ? (100 - ((stats.errorCount / stats.totalLogs) * 100)).toFixed(2) 
                  : '100.00'}%
              </p>
              <p className="text-sm text-slate-500 mt-1">Health Score</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
