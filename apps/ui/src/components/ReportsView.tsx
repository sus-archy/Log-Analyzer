'use client';

import { useState, useEffect, useRef } from 'react';
import {
  FileText, Download, RefreshCw, Loader2, AlertTriangle,
  Shield, Activity, Server, TrendingUp, CheckCircle,
  XCircle, AlertCircle, Clock, BarChart3, PieChart
} from 'lucide-react';
import {
  getFullReport, getTimeRange,
  type FullReport, type PerformanceMetrics, type SecurityMetrics, type ServiceHealth
} from '@/lib/api';

interface Props {
  serviceName: string;
  timeRange: string;
}

// Simple bar chart component (no external lib needed)
function SimpleBarChart({ 
  data, 
  labelKey, 
  valueKey, 
  color = 'bg-blue-500',
  maxHeight = 200 
}: { 
  data: Array<Record<string, any>>; 
  labelKey: string; 
  valueKey: string;
  color?: string;
  maxHeight?: number;
}) {
  const maxValue = Math.max(...data.map(d => d[valueKey] || 0), 1);
  
  return (
    <div className="flex items-end gap-1 h-full" style={{ height: maxHeight }}>
      {data.map((item, idx) => {
        const height = (item[valueKey] / maxValue) * maxHeight;
        return (
          <div key={idx} className="flex-1 flex flex-col items-center justify-end">
            <div 
              className={`w-full ${color} rounded-t transition-all duration-300 hover:opacity-80`}
              style={{ height: Math.max(2, height) }}
              title={`${item[labelKey]}: ${item[valueKey].toLocaleString()}`}
            />
            <span className="text-[10px] text-slate-500 mt-1 truncate w-full text-center">
              {item[labelKey]?.slice(-5) || ''}
            </span>
          </div>
        );
      })}
    </div>
  );
}

// Pie chart component using CSS
function SimplePieChart({ 
  data 
}: { 
  data: Array<{ label: string; value: number; color: string }> 
}) {
  const total = data.reduce((sum, d) => sum + d.value, 0);
  if (total === 0) return <div className="text-slate-400 text-center">No data</div>;
  
  let cumulativePercent = 0;
  const segments = data.map(d => {
    const percent = (d.value / total) * 100;
    const segment = {
      ...d,
      percent,
      offset: cumulativePercent,
    };
    cumulativePercent += percent;
    return segment;
  });
  
  // Create conic gradient
  const gradient = segments
    .filter(s => s.percent > 0)
    .map(s => `${s.color} ${s.offset}% ${s.offset + s.percent}%`)
    .join(', ');
  
  return (
    <div className="flex items-center gap-4">
      <div 
        className="w-32 h-32 rounded-full shadow-inner"
        style={{ background: `conic-gradient(${gradient})` }}
      />
      <div className="space-y-1">
        {segments.filter(s => s.value > 0).map((s, idx) => (
          <div key={idx} className="flex items-center gap-2 text-sm">
            <div className="w-3 h-3 rounded" style={{ background: s.color }} />
            <span className="text-slate-600">{s.label}:</span>
            <span className="font-medium">{s.value.toLocaleString()}</span>
            <span className="text-slate-400">({s.percent.toFixed(1)}%)</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// Gauge chart for security score
function GaugeChart({ score, riskLevel }: { score: number; riskLevel: string }) {
  const getColor = () => {
    if (score >= 90) return { ring: 'stroke-green-500', text: 'text-green-600', bg: 'bg-green-100' };
    if (score >= 70) return { ring: 'stroke-yellow-500', text: 'text-yellow-600', bg: 'bg-yellow-100' };
    if (score >= 50) return { ring: 'stroke-orange-500', text: 'text-orange-600', bg: 'bg-orange-100' };
    return { ring: 'stroke-red-500', text: 'text-red-600', bg: 'bg-red-100' };
  };
  
  const colors = getColor();
  const circumference = 2 * Math.PI * 45;
  const strokeDashoffset = circumference - (score / 100) * circumference;
  
  return (
    <div className="flex flex-col items-center">
      <svg className="w-36 h-36 transform -rotate-90">
        <circle
          cx="72"
          cy="72"
          r="45"
          fill="none"
          stroke="#e2e8f0"
          strokeWidth="10"
        />
        <circle
          cx="72"
          cy="72"
          r="45"
          fill="none"
          className={colors.ring}
          strokeWidth="10"
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          strokeLinecap="round"
        />
      </svg>
      <div className="absolute flex flex-col items-center justify-center w-36 h-36">
        <span className={`text-3xl font-bold ${colors.text}`}>{score}</span>
        <span className="text-sm text-slate-500">/ 100</span>
      </div>
      <span className={`mt-2 px-3 py-1 rounded-full text-sm font-medium ${colors.bg} ${colors.text}`}>
        {riskLevel} Risk
      </span>
    </div>
  );
}

export default function ReportsView({ serviceName, timeRange }: Props) {
  const [report, setReport] = useState<FullReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'overview' | 'security' | 'services'>('overview');
  const reportRef = useRef<HTMLDivElement>(null);

  const loadReport = async () => {
    setLoading(true);
    setError(null);

    try {
      const { from, to } = getTimeRange(timeRange);
      const data = await getFullReport({
        service_name: serviceName || undefined,
        from,
        to,
      });
      setReport(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load report');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadReport();
  }, [serviceName, timeRange]);

  const exportToPDF = () => {
    // Simple print-based export
    if (reportRef.current) {
      const printWindow = window.open('', '_blank');
      if (printWindow) {
        printWindow.document.write(`
          <html>
            <head>
              <title>LogMind AI Report - ${new Date().toLocaleDateString()}</title>
              <style>
                body { font-family: Arial, sans-serif; padding: 20px; }
                h1 { color: #1e293b; }
                h2 { color: #475569; border-bottom: 1px solid #e2e8f0; padding-bottom: 8px; }
                .metric { display: inline-block; margin: 10px; padding: 15px; background: #f8fafc; border-radius: 8px; }
                .metric-value { font-size: 24px; font-weight: bold; color: #1e293b; }
                .metric-label { font-size: 12px; color: #64748b; }
                table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                th, td { padding: 10px; border: 1px solid #e2e8f0; text-align: left; }
                th { background: #f1f5f9; }
                .warning { color: #f59e0b; }
                .error { color: #ef4444; }
                .success { color: #10b981; }
              </style>
            </head>
            <body>
              <h1>🔍 LogMind AI - Log Analysis Report</h1>
              <p><strong>Generated:</strong> ${new Date().toLocaleString()}</p>
              <p><strong>Service:</strong> ${serviceName || 'All Services'}</p>
              <p><strong>Time Range:</strong> ${timeRange}</p>
              
              <h2>📊 Performance Metrics</h2>
              <div class="metric">
                <div class="metric-value">${report?.performance.total_logs.toLocaleString()}</div>
                <div class="metric-label">Total Logs</div>
              </div>
              <div class="metric">
                <div class="metric-value">${report?.performance.total_templates.toLocaleString()}</div>
                <div class="metric-label">Templates</div>
              </div>
              <div class="metric">
                <div class="metric-value">${report?.performance.error_rate}%</div>
                <div class="metric-label">Error Rate</div>
              </div>
              <div class="metric">
                <div class="metric-value">${report?.performance.services_count}</div>
                <div class="metric-label">Services</div>
              </div>
              
              <h2>🛡️ Security Assessment</h2>
              <div class="metric">
                <div class="metric-value ${report?.security.security_score && report.security.security_score >= 70 ? 'success' : 'error'}">
                  ${report?.security.security_score}/100
                </div>
                <div class="metric-label">Security Score</div>
              </div>
              <div class="metric">
                <div class="metric-value">${report?.security.risk_level}</div>
                <div class="metric-label">Risk Level</div>
              </div>
              
              <h3>Security Events by Category</h3>
              <table>
                <tr>
                  <th>Category</th>
                  <th>Count</th>
                </tr>
                ${Object.entries(report?.security.categories || {}).map(([key, value]) => `
                  <tr>
                    <td>${key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</td>
                    <td>${value}</td>
                  </tr>
                `).join('')}
              </table>
              
              <h3>Recommendations</h3>
              <ul>
                ${report?.security.recommendations.map(r => `<li>${r}</li>`).join('')}
              </ul>
              
              <h2>🖥️ Services Health</h2>
              <table>
                <tr>
                  <th>Service</th>
                  <th>Status</th>
                  <th>Logs</th>
                  <th>Errors</th>
                  <th>Error Rate</th>
                </tr>
                ${report?.services_health.slice(0, 20).map(s => `
                  <tr>
                    <td>${s.service_name}</td>
                    <td class="${s.status === 'healthy' ? 'success' : s.status === 'critical' ? 'error' : 'warning'}">${s.status.toUpperCase()}</td>
                    <td>${s.total_logs.toLocaleString()}</td>
                    <td>${s.errors.toLocaleString()}</td>
                    <td>${s.error_rate}%</td>
                  </tr>
                `).join('')}
              </table>
              
              <hr>
              <p style="color: #64748b; font-size: 12px;">
                Report generated by LogMind AI • ${new Date().toISOString()}
              </p>
            </body>
          </html>
        `);
        printWindow.document.close();
        printWindow.print();
      }
    }
  };

  if (loading) {
    return (
      <div className="h-full flex items-center justify-center bg-white">
        <div className="text-center">
          <Loader2 size={48} className="animate-spin mx-auto text-blue-500 mb-4" />
          <p className="text-slate-600 font-medium">Generating report...</p>
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
            onClick={loadReport}
            className="mt-4 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 flex items-center gap-2 mx-auto"
          >
            <RefreshCw size={16} /> Retry
          </button>
        </div>
      </div>
    );
  }

  if (!report) return null;

  const { performance: perf, security: sec, services_health: services } = report;

  const severityPieData = [
    { label: 'Critical', value: perf.severity_breakdown.critical, color: '#dc2626' },
    { label: 'Error', value: perf.severity_breakdown.error, color: '#f97316' },
    { label: 'Warning', value: perf.severity_breakdown.warning, color: '#eab308' },
    { label: 'Info', value: perf.severity_breakdown.info, color: '#3b82f6' },
    { label: 'Debug', value: perf.severity_breakdown.debug, color: '#94a3b8' },
  ];

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy': return <CheckCircle className="text-green-500" size={18} />;
      case 'warning': return <AlertCircle className="text-yellow-500" size={18} />;
      case 'degraded': return <AlertTriangle className="text-orange-500" size={18} />;
      case 'critical': return <XCircle className="text-red-500" size={18} />;
      default: return null;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'bg-green-100 text-green-700';
      case 'warning': return 'bg-yellow-100 text-yellow-700';
      case 'degraded': return 'bg-orange-100 text-orange-700';
      case 'critical': return 'bg-red-100 text-red-700';
      default: return 'bg-slate-100 text-slate-700';
    }
  };

  return (
    <div ref={reportRef} className="h-full overflow-auto bg-slate-50 p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-slate-900 flex items-center gap-2">
              <FileText className="text-blue-500" />
              Log Analysis Report
            </h1>
            <p className="text-slate-500 mt-1">
              {serviceName ? `Service: ${serviceName}` : 'All Services'} • {timeRange} • 
              Generated: {new Date(report.generated_at).toLocaleString()}
            </p>
          </div>
          <div className="flex gap-2">
            <button
              onClick={loadReport}
              className="px-4 py-2 bg-white border border-slate-200 rounded-lg hover:bg-slate-50 flex items-center gap-2 text-slate-700 shadow-sm"
            >
              <RefreshCw size={16} /> Refresh
            </button>
            <button
              onClick={exportToPDF}
              className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 flex items-center gap-2 shadow-sm"
            >
              <Download size={16} /> Export PDF
            </button>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="flex gap-2 border-b border-slate-200">
          {[
            { id: 'overview', label: 'Overview', icon: BarChart3 },
            { id: 'security', label: 'Security', icon: Shield },
            { id: 'services', label: 'Services Health', icon: Server },
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`px-4 py-3 flex items-center gap-2 font-medium transition-colors border-b-2 -mb-px ${
                activeTab === tab.id
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-slate-500 hover:text-slate-700'
              }`}
            >
              <tab.icon size={18} />
              {tab.label}
            </button>
          ))}
        </div>

        {/* Overview Tab */}
        {activeTab === 'overview' && (
          <div className="space-y-6">
            {/* Summary Cards */}
            <div className="grid grid-cols-5 gap-4">
              <div className="bg-white rounded-xl p-5 border border-slate-200 shadow-sm">
                <div className="flex items-center gap-2 text-slate-500 text-sm mb-2">
                  <Activity size={16} /> Total Logs
                </div>
                <p className="text-3xl font-bold text-slate-900">{perf.total_logs.toLocaleString()}</p>
              </div>
              <div className="bg-white rounded-xl p-5 border border-slate-200 shadow-sm">
                <div className="flex items-center gap-2 text-slate-500 text-sm mb-2">
                  <FileText size={16} /> Templates
                </div>
                <p className="text-3xl font-bold text-slate-900">{perf.total_templates.toLocaleString()}</p>
              </div>
              <div className="bg-white rounded-xl p-5 border border-slate-200 shadow-sm">
                <div className="flex items-center gap-2 text-slate-500 text-sm mb-2">
                  <Server size={16} /> Services
                </div>
                <p className="text-3xl font-bold text-slate-900">{perf.services_count}</p>
              </div>
              <div className="bg-white rounded-xl p-5 border border-slate-200 shadow-sm">
                <div className="flex items-center gap-2 text-red-500 text-sm mb-2">
                  <AlertTriangle size={16} /> Error Rate
                </div>
                <p className="text-3xl font-bold text-red-600">{perf.error_rate}%</p>
              </div>
              <div className="bg-white rounded-xl p-5 border border-slate-200 shadow-sm">
                <div className="flex items-center gap-2 text-slate-500 text-sm mb-2">
                  <TrendingUp size={16} /> AI Coverage
                </div>
                <p className="text-3xl font-bold text-blue-600">{perf.embedding_coverage}%</p>
              </div>
            </div>

            {/* Charts Row */}
            <div className="grid grid-cols-2 gap-6">
              {/* Severity Distribution */}
              <div className="bg-white rounded-xl p-6 border border-slate-200 shadow-sm">
                <h3 className="font-semibold text-slate-800 mb-4 flex items-center gap-2">
                  <PieChart size={18} className="text-blue-500" />
                  Severity Distribution
                </h3>
                <SimplePieChart data={severityPieData} />
              </div>

              {/* Hourly Trend */}
              <div className="bg-white rounded-xl p-6 border border-slate-200 shadow-sm">
                <h3 className="font-semibold text-slate-800 mb-4 flex items-center gap-2">
                  <TrendingUp size={18} className="text-blue-500" />
                  Log Volume (Hourly Trend)
                </h3>
                {perf.hourly_trend.length > 0 ? (
                  <SimpleBarChart 
                    data={perf.hourly_trend} 
                    labelKey="hour" 
                    valueKey="count"
                    color="bg-blue-500"
                    maxHeight={180}
                  />
                ) : (
                  <div className="h-44 flex items-center justify-center text-slate-400">
                    No trend data available
                  </div>
                )}
              </div>
            </div>

            {/* Template Efficiency */}
            <div className="bg-white rounded-xl p-6 border border-slate-200 shadow-sm">
              <h3 className="font-semibold text-slate-800 mb-4">Performance Insights</h3>
              <div className="grid grid-cols-3 gap-6">
                <div className="text-center p-4 bg-slate-50 rounded-lg">
                  <p className="text-4xl font-bold text-blue-600">{perf.template_efficiency.toFixed(0)}</p>
                  <p className="text-sm text-slate-500 mt-1">Avg Logs per Template</p>
                  <p className="text-xs text-slate-400">Higher = more log pattern reuse</p>
                </div>
                <div className="text-center p-4 bg-slate-50 rounded-lg">
                  <p className="text-4xl font-bold text-green-600">{perf.embedded_templates.toLocaleString()}</p>
                  <p className="text-sm text-slate-500 mt-1">AI-Indexed Templates</p>
                  <p className="text-xs text-slate-400">Ready for semantic search</p>
                </div>
                <div className="text-center p-4 bg-slate-50 rounded-lg">
                  <p className="text-4xl font-bold text-amber-600">{perf.warning_rate}%</p>
                  <p className="text-sm text-slate-500 mt-1">Warning Rate</p>
                  <p className="text-xs text-slate-400">Non-critical issues</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Security Tab */}
        {activeTab === 'security' && (
          <div className="space-y-6">
            {/* Security Score */}
            <div className="grid grid-cols-3 gap-6">
              <div className="bg-white rounded-xl p-6 border border-slate-200 shadow-sm flex flex-col items-center justify-center relative">
                <h3 className="font-semibold text-slate-800 mb-4">Security Score</h3>
                <GaugeChart score={sec.security_score} riskLevel={sec.risk_level} />
              </div>
              
              {/* Security Categories */}
              <div className="col-span-2 bg-white rounded-xl p-6 border border-slate-200 shadow-sm">
                <h3 className="font-semibold text-slate-800 mb-4">Security Events by Category</h3>
                <div className="space-y-3">
                  {Object.entries(sec.categories).map(([category, count]) => {
                    const maxCount = Math.max(...Object.values(sec.categories), 1);
                    const percentage = (count / maxCount) * 100;
                    return (
                      <div key={category}>
                        <div className="flex justify-between text-sm mb-1">
                          <span className="text-slate-600 capitalize">
                            {category.replace(/_/g, ' ')}
                          </span>
                          <span className="font-medium text-slate-800">{count.toLocaleString()}</span>
                        </div>
                        <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                          <div 
                            className={`h-full rounded-full transition-all duration-500 ${
                              count > 10 ? 'bg-red-500' : count > 5 ? 'bg-orange-500' : 'bg-blue-500'
                            }`}
                            style={{ width: `${percentage}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* Recommendations */}
            <div className="bg-white rounded-xl p-6 border border-slate-200 shadow-sm">
              <h3 className="font-semibold text-slate-800 mb-4 flex items-center gap-2">
                <Shield size={18} className="text-blue-500" />
                Security Recommendations
              </h3>
              <div className="space-y-3">
                {sec.recommendations.map((rec, idx) => (
                  <div 
                    key={idx}
                    className={`p-4 rounded-lg flex items-start gap-3 ${
                      rec.includes('CRITICAL') ? 'bg-red-50 border border-red-200' :
                      rec.includes('✅') ? 'bg-green-50 border border-green-200' :
                      'bg-amber-50 border border-amber-200'
                    }`}
                  >
                    {rec.includes('CRITICAL') ? (
                      <XCircle className="text-red-500 mt-0.5 flex-shrink-0" size={20} />
                    ) : rec.includes('✅') ? (
                      <CheckCircle className="text-green-500 mt-0.5 flex-shrink-0" size={20} />
                    ) : (
                      <AlertCircle className="text-amber-500 mt-0.5 flex-shrink-0" size={20} />
                    )}
                    <span className="text-slate-700">{rec}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Top Affected Services */}
            <div className="bg-white rounded-xl p-6 border border-slate-200 shadow-sm">
              <h3 className="font-semibold text-slate-800 mb-4">Top Affected Services</h3>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-slate-200">
                      <th className="text-left py-3 px-4 font-medium text-slate-600">Service</th>
                      <th className="text-right py-3 px-4 font-medium text-slate-600">Error Count</th>
                      <th className="text-right py-3 px-4 font-medium text-slate-600">Impact</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sec.top_affected_services.slice(0, 10).map((service, idx) => (
                      <tr key={idx} className="border-b border-slate-100 hover:bg-slate-50">
                        <td className="py-3 px-4 font-medium text-slate-800">{service.service}</td>
                        <td className="py-3 px-4 text-right text-red-600 font-medium">
                          {service.error_count.toLocaleString()}
                        </td>
                        <td className="py-3 px-4 text-right">
                          <div className="w-24 h-2 bg-slate-100 rounded-full overflow-hidden ml-auto">
                            <div 
                              className="h-full bg-red-500 rounded-full"
                              style={{ 
                                width: `${Math.min(100, (service.error_count / (sec.top_affected_services[0]?.error_count || 1)) * 100)}%` 
                              }}
                            />
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* Services Health Tab */}
        {activeTab === 'services' && (
          <div className="space-y-6">
            {/* Status Summary */}
            <div className="grid grid-cols-4 gap-4">
              {['healthy', 'warning', 'degraded', 'critical'].map(status => {
                const count = services.filter(s => s.status === status).length;
                return (
                  <div key={status} className={`rounded-xl p-5 border shadow-sm ${
                    status === 'healthy' ? 'bg-green-50 border-green-200' :
                    status === 'warning' ? 'bg-yellow-50 border-yellow-200' :
                    status === 'degraded' ? 'bg-orange-50 border-orange-200' :
                    'bg-red-50 border-red-200'
                  }`}>
                    <div className="flex items-center gap-2 mb-2">
                      {getStatusIcon(status)}
                      <span className="font-medium capitalize">{status}</span>
                    </div>
                    <p className="text-3xl font-bold">{count}</p>
                    <p className="text-sm text-slate-500">services</p>
                  </div>
                );
              })}
            </div>

            {/* Services Table */}
            <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
              <table className="w-full">
                <thead className="bg-slate-50">
                  <tr>
                    <th className="text-left py-3 px-4 font-medium text-slate-600">Service</th>
                    <th className="text-center py-3 px-4 font-medium text-slate-600">Status</th>
                    <th className="text-right py-3 px-4 font-medium text-slate-600">Total Logs</th>
                    <th className="text-right py-3 px-4 font-medium text-slate-600">Errors</th>
                    <th className="text-right py-3 px-4 font-medium text-slate-600">Warnings</th>
                    <th className="text-right py-3 px-4 font-medium text-slate-600">Error Rate</th>
                    <th className="text-right py-3 px-4 font-medium text-slate-600">Last Seen</th>
                  </tr>
                </thead>
                <tbody>
                  {services.map((service, idx) => (
                    <tr key={idx} className="border-b border-slate-100 hover:bg-slate-50">
                      <td className="py-3 px-4 font-medium text-slate-800">{service.service_name}</td>
                      <td className="py-3 px-4 text-center">
                        <span className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(service.status)}`}>
                          {getStatusIcon(service.status)}
                          {service.status.toUpperCase()}
                        </span>
                      </td>
                      <td className="py-3 px-4 text-right text-slate-600">
                        {service.total_logs.toLocaleString()}
                      </td>
                      <td className="py-3 px-4 text-right text-red-600 font-medium">
                        {service.errors.toLocaleString()}
                      </td>
                      <td className="py-3 px-4 text-right text-amber-600">
                        {service.warnings.toLocaleString()}
                      </td>
                      <td className="py-3 px-4 text-right">
                        <span className={`font-medium ${
                          service.error_rate >= 10 ? 'text-red-600' :
                          service.error_rate >= 5 ? 'text-orange-600' :
                          service.error_rate >= 1 ? 'text-yellow-600' :
                          'text-green-600'
                        }`}>
                          {service.error_rate}%
                        </span>
                      </td>
                      <td className="py-3 px-4 text-right text-slate-400 text-sm">
                        {new Date(service.last_seen).toLocaleString()}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
