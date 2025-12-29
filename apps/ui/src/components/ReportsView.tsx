'use client';

import { useState, useEffect, useRef } from 'react';
import {
  FileText, Download, RefreshCw, Loader2, AlertTriangle,
  Shield, Activity, Server, TrendingUp, CheckCircle,
  XCircle, AlertCircle, Clock, Database, Zap, Eye,
  ArrowUpRight, ArrowDownRight, Minus, Layers, Lock,
  Unlock, Bug, Globe, Wifi, HardDrive, Cpu
} from 'lucide-react';
import {
  getFullReport, getTimeRange,
  type FullReport, type PerformanceMetrics, type SecurityMetrics, type ServiceHealth
} from '@/lib/api';

interface Props {
  serviceName: string;
  timeRange: string;
}

// Stat Card Component
function StatCard({ 
  icon: Icon, 
  label, 
  value, 
  subValue,
  trend,
  color = 'blue',
  size = 'normal'
}: { 
  icon: any; 
  label: string; 
  value: string | number; 
  subValue?: string;
  trend?: 'up' | 'down' | 'neutral';
  color?: 'blue' | 'green' | 'red' | 'amber' | 'purple' | 'slate';
  size?: 'normal' | 'large';
}) {
  const colorClasses = {
    blue: 'bg-blue-50 text-blue-600 border-blue-100',
    green: 'bg-emerald-50 text-emerald-600 border-emerald-100',
    red: 'bg-red-50 text-red-600 border-red-100',
    amber: 'bg-amber-50 text-amber-600 border-amber-100',
    purple: 'bg-purple-50 text-purple-600 border-purple-100',
    slate: 'bg-slate-50 text-slate-600 border-slate-100',
  };
  
  const iconColors = {
    blue: 'text-blue-500 bg-blue-100',
    green: 'text-emerald-500 bg-emerald-100',
    red: 'text-red-500 bg-red-100',
    amber: 'text-amber-500 bg-amber-100',
    purple: 'text-purple-500 bg-purple-100',
    slate: 'text-slate-500 bg-slate-100',
  };

  return (
    <div className={`bg-white rounded-2xl p-5 border border-slate-100 shadow-sm hover:shadow-md transition-shadow ${size === 'large' ? 'p-6' : ''}`}>
      <div className="flex items-start justify-between">
        <div className={`p-2.5 rounded-xl ${iconColors[color]}`}>
          <Icon size={size === 'large' ? 24 : 20} />
        </div>
        {trend && (
          <div className={`flex items-center gap-1 text-xs font-medium px-2 py-1 rounded-full ${
            trend === 'up' ? 'bg-emerald-50 text-emerald-600' :
            trend === 'down' ? 'bg-red-50 text-red-600' :
            'bg-slate-50 text-slate-500'
          }`}>
            {trend === 'up' ? <ArrowUpRight size={12} /> : trend === 'down' ? <ArrowDownRight size={12} /> : <Minus size={12} />}
            {trend === 'up' ? 'Good' : trend === 'down' ? 'Alert' : 'Stable'}
          </div>
        )}
      </div>
      <div className="mt-4">
        <p className={`font-bold text-slate-900 ${size === 'large' ? 'text-4xl' : 'text-2xl'}`}>
          {typeof value === 'number' ? value.toLocaleString() : value}
        </p>
        <p className="text-slate-500 text-sm mt-1">{label}</p>
        {subValue && <p className="text-slate-400 text-xs mt-0.5">{subValue}</p>}
      </div>
    </div>
  );
}

// Severity Badge
function SeverityBadge({ level, count }: { level: string; count: number }) {
  const config: Record<string, { bg: string; text: string; icon: any }> = {
    critical: { bg: 'bg-red-500', text: 'text-white', icon: XCircle },
    error: { bg: 'bg-orange-500', text: 'text-white', icon: AlertTriangle },
    warning: { bg: 'bg-amber-400', text: 'text-amber-900', icon: AlertCircle },
    info: { bg: 'bg-blue-500', text: 'text-white', icon: Eye },
    debug: { bg: 'bg-slate-400', text: 'text-white', icon: Bug },
  };
  
  const { bg, text, icon: Icon } = config[level] || config.info;
  
  return (
    <div className={`${bg} ${text} rounded-xl p-4 flex items-center gap-3`}>
      <Icon size={24} />
      <div>
        <p className="text-2xl font-bold">{count.toLocaleString()}</p>
        <p className="text-sm opacity-80 capitalize">{level}</p>
      </div>
    </div>
  );
}

// Progress Ring
function ProgressRing({ value, size = 120, color = '#3b82f6' }: { value: number; size?: number; color?: string }) {
  const strokeWidth = 10;
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (value / 100) * circumference;
  
  return (
    <div className="relative inline-flex items-center justify-center">
      <svg width={size} height={size} className="transform -rotate-90">
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="#e2e8f0"
          strokeWidth={strokeWidth}
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={strokeWidth}
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          className="transition-all duration-1000"
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-3xl font-bold text-slate-800">{value}</span>
        <span className="text-xs text-slate-400">/ 100</span>
      </div>
    </div>
  );
}

// Status Indicator
function StatusIndicator({ status }: { status: string }) {
  const config: Record<string, { color: string; bg: string; pulse: boolean }> = {
    healthy: { color: 'bg-emerald-500', bg: 'bg-emerald-100', pulse: false },
    warning: { color: 'bg-amber-500', bg: 'bg-amber-100', pulse: true },
    degraded: { color: 'bg-orange-500', bg: 'bg-orange-100', pulse: true },
    critical: { color: 'bg-red-500', bg: 'bg-red-100', pulse: true },
  };
  
  const { color, bg, pulse } = config[status] || config.healthy;
  
  return (
    <div className={`relative flex items-center gap-2 px-3 py-1.5 rounded-full ${bg}`}>
      <div className={`w-2 h-2 rounded-full ${color} ${pulse ? 'animate-pulse' : ''}`} />
      <span className="text-xs font-medium capitalize text-slate-700">{status}</span>
    </div>
  );
}

// Service Row
function ServiceRow({ service, maxLogs }: { service: ServiceHealth; maxLogs: number }) {
  const barWidth = (service.total_logs / maxLogs) * 100;
  const errorBarWidth = service.errors > 0 ? (service.errors / service.total_logs) * 100 : 0;
  
  return (
    <div className="group bg-white rounded-xl p-4 border border-slate-100 hover:border-slate-200 hover:shadow-sm transition-all">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-slate-100 to-slate-50 flex items-center justify-center">
            <Server size={18} className="text-slate-500" />
          </div>
          <div>
            <p className="font-semibold text-slate-800">{service.service_name}</p>
            <p className="text-xs text-slate-400">{service.total_logs.toLocaleString()} logs</p>
          </div>
        </div>
        <StatusIndicator status={service.status} />
      </div>
      
      {/* Visual bar */}
      <div className="h-2 bg-slate-100 rounded-full overflow-hidden mb-2">
        <div 
          className="h-full bg-gradient-to-r from-blue-400 to-blue-500 rounded-full relative"
          style={{ width: `${barWidth}%` }}
        >
          {errorBarWidth > 0 && (
            <div 
              className="absolute right-0 top-0 h-full bg-red-500 rounded-r-full"
              style={{ width: `${errorBarWidth}%` }}
            />
          )}
        </div>
      </div>
      
      {/* Stats row */}
      <div className="flex items-center gap-4 text-xs text-slate-500">
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-red-500" />
          {service.errors.toLocaleString()} errors
        </span>
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-amber-400" />
          {service.warnings.toLocaleString()} warnings
        </span>
        <span className="ml-auto font-medium text-slate-600">
          {service.error_rate}% error rate
        </span>
      </div>
    </div>
  );
}

// Security Threat Card
function ThreatCard({ category, count, maxCount }: { category: string; count: number; maxCount: number }) {
  const percentage = (count / maxCount) * 100;
  const isHigh = count > maxCount * 0.5;
  const isMedium = count > maxCount * 0.2;
  
  const icons: Record<string, any> = {
    authentication_failures: Lock,
    brute_force_indicators: Unlock,
    suspicious_access: Eye,
    network_issues: Wifi,
    system_errors: Cpu,
  };
  
  const Icon = icons[category] || AlertTriangle;
  
  return (
    <div className={`rounded-xl p-4 border transition-all ${
      isHigh ? 'bg-red-50 border-red-200' : 
      isMedium ? 'bg-amber-50 border-amber-200' : 
      'bg-slate-50 border-slate-200'
    }`}>
      <div className="flex items-center gap-3 mb-3">
        <div className={`p-2 rounded-lg ${
          isHigh ? 'bg-red-100 text-red-600' : 
          isMedium ? 'bg-amber-100 text-amber-600' : 
          'bg-slate-100 text-slate-600'
        }`}>
          <Icon size={18} />
        </div>
        <div className="flex-1">
          <p className="text-sm font-medium text-slate-700 capitalize">
            {category.replace(/_/g, ' ')}
          </p>
        </div>
        <p className={`text-xl font-bold ${
          isHigh ? 'text-red-600' : isMedium ? 'text-amber-600' : 'text-slate-600'
        }`}>
          {count.toLocaleString()}
        </p>
      </div>
      <div className="h-1.5 bg-white/50 rounded-full overflow-hidden">
        <div 
          className={`h-full rounded-full transition-all duration-500 ${
            isHigh ? 'bg-red-500' : isMedium ? 'bg-amber-500' : 'bg-slate-400'
          }`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}

// Recommendation Card
function RecommendationCard({ text, index }: { text: string; index: number }) {
  const isCritical = text.includes('CRITICAL');
  const isSuccess = text.includes('✅');
  
  return (
    <div className={`flex gap-4 p-4 rounded-xl border ${
      isCritical ? 'bg-red-50 border-red-200' :
      isSuccess ? 'bg-emerald-50 border-emerald-200' :
      'bg-amber-50 border-amber-200'
    }`}>
      <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold ${
        isCritical ? 'bg-red-500 text-white' :
        isSuccess ? 'bg-emerald-500 text-white' :
        'bg-amber-500 text-white'
      }`}>
        {isSuccess ? <CheckCircle size={16} /> : index + 1}
      </div>
      <div className="flex-1">
        <p className="text-slate-700 text-sm leading-relaxed">{text}</p>
      </div>
    </div>
  );
}

// Hourly Activity Grid (replaces bar chart)
function ActivityGrid({ data }: { data: Array<{ hour: string; count: number; errors: number }> }) {
  if (!data || data.length === 0) {
    return <div className="text-center text-slate-400 py-8">No activity data available</div>;
  }
  
  const maxCount = Math.max(...data.map(d => d.count), 1);
  
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-12 gap-2">
        {data.slice(-24).map((item, idx) => {
          const intensity = item.count / maxCount;
          const hasErrors = item.errors > 0;
          const hour = item.hour?.slice(11, 13) || idx.toString();
          
          return (
            <div key={idx} className="flex flex-col items-center gap-1 group relative">
              {/* Tooltip */}
              <div className="absolute bottom-full mb-2 hidden group-hover:block z-10 pointer-events-none">
                <div className="bg-slate-800 text-white text-xs rounded-lg px-3 py-2 whitespace-nowrap shadow-lg">
                  <p className="font-semibold">{item.count.toLocaleString()} logs</p>
                  {hasErrors && <p className="text-red-300">{item.errors} errors</p>}
                  <p className="text-slate-400 text-[10px] mt-1">{item.hour?.slice(0, 16)}</p>
                </div>
              </div>
              {/* Activity block */}
              <div 
                className={`w-full aspect-square rounded-lg transition-all cursor-pointer hover:scale-110 ${
                  hasErrors 
                    ? 'bg-gradient-to-br from-red-400 to-red-500 shadow-red-500/30 shadow-sm' 
                    : intensity > 0.7 
                      ? 'bg-gradient-to-br from-blue-500 to-blue-600 shadow-blue-500/30 shadow-sm'
                      : intensity > 0.4 
                        ? 'bg-blue-400'
                        : intensity > 0.1 
                          ? 'bg-blue-300'
                          : 'bg-slate-200'
                }`}
                style={{ opacity: 0.3 + intensity * 0.7 }}
              />
              <span className="text-[10px] text-slate-400">{hour}h</span>
            </div>
          );
        })}
      </div>
      
      {/* Legend */}
      <div className="flex items-center justify-center gap-6 text-xs text-slate-500">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded bg-slate-200" />
          <span>Low</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded bg-blue-400" />
          <span>Medium</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded bg-blue-600" />
          <span>High</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded bg-red-500" />
          <span>Errors</span>
        </div>
      </div>
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
    if (reportRef.current) {
      const printWindow = window.open('', '_blank');
      if (printWindow && report) {
        printWindow.document.write(`
          <html>
            <head>
              <title>LogMind AI Report - ${new Date().toLocaleDateString()}</title>
              <style>
                body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; padding: 40px; color: #1e293b; }
                h1 { color: #0f172a; margin-bottom: 8px; }
                h2 { color: #334155; border-bottom: 2px solid #e2e8f0; padding-bottom: 12px; margin-top: 32px; }
                .header-info { color: #64748b; margin-bottom: 32px; }
                .stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 24px 0; }
                .stat-card { background: #f8fafc; border-radius: 12px; padding: 20px; }
                .stat-value { font-size: 28px; font-weight: 700; color: #0f172a; }
                .stat-label { font-size: 13px; color: #64748b; margin-top: 4px; }
                table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                th, td { padding: 12px; text-align: left; border-bottom: 1px solid #e2e8f0; }
                th { background: #f1f5f9; font-weight: 600; }
                .badge { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 500; }
                .badge-healthy { background: #dcfce7; color: #166534; }
                .badge-warning { background: #fef9c3; color: #854d0e; }
                .badge-critical { background: #fee2e2; color: #991b1b; }
                .recommendation { padding: 16px; margin: 12px 0; border-radius: 8px; border-left: 4px solid; }
                .rec-warning { background: #fffbeb; border-color: #f59e0b; }
                .rec-success { background: #ecfdf5; border-color: #10b981; }
                .rec-critical { background: #fef2f2; border-color: #ef4444; }
                footer { margin-top: 40px; padding-top: 20px; border-top: 1px solid #e2e8f0; color: #94a3b8; font-size: 12px; }
              </style>
            </head>
            <body>
              <h1>🔍 LogMind AI - Log Analysis Report</h1>
              <div class="header-info">
                <p><strong>Generated:</strong> ${new Date().toLocaleString()}</p>
                <p><strong>Service:</strong> ${serviceName || 'All Services'} &nbsp;|&nbsp; <strong>Period:</strong> ${timeRange}</p>
              </div>
              
              <h2>📊 Performance Overview</h2>
              <div class="stats-grid">
                <div class="stat-card">
                  <div class="stat-value">${report.performance.total_logs.toLocaleString()}</div>
                  <div class="stat-label">Total Logs</div>
                </div>
                <div class="stat-card">
                  <div class="stat-value">${report.performance.total_templates.toLocaleString()}</div>
                  <div class="stat-label">Templates</div>
                </div>
                <div class="stat-card">
                  <div class="stat-value">${report.performance.error_rate}%</div>
                  <div class="stat-label">Error Rate</div>
                </div>
                <div class="stat-card">
                  <div class="stat-value">${report.performance.services_count}</div>
                  <div class="stat-label">Services</div>
                </div>
              </div>
              
              <h2>🛡️ Security Assessment</h2>
              <div class="stats-grid" style="grid-template-columns: repeat(2, 1fr);">
                <div class="stat-card">
                  <div class="stat-value" style="color: ${report.security.security_score >= 70 ? '#16a34a' : '#dc2626'}">${report.security.security_score}/100</div>
                  <div class="stat-label">Security Score</div>
                </div>
                <div class="stat-card">
                  <div class="stat-value">${report.security.risk_level}</div>
                  <div class="stat-label">Risk Level</div>
                </div>
              </div>
              
              <h2>🚨 Security Events</h2>
              <table>
                <tr><th>Category</th><th>Count</th></tr>
                ${Object.entries(report.security.categories).map(([k, v]) => `
                  <tr><td>${k.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</td><td>${v}</td></tr>
                `).join('')}
              </table>
              
              <h2>💡 Recommendations</h2>
              ${report.security.recommendations.map(r => `
                <div class="recommendation ${r.includes('CRITICAL') ? 'rec-critical' : r.includes('✅') ? 'rec-success' : 'rec-warning'}">${r}</div>
              `).join('')}
              
              <h2>🖥️ Services Health</h2>
              <table>
                <tr><th>Service</th><th>Status</th><th>Logs</th><th>Errors</th><th>Error Rate</th></tr>
                ${report.services_health.slice(0, 15).map(s => `
                  <tr>
                    <td>${s.service_name}</td>
                    <td><span class="badge badge-${s.status === 'healthy' ? 'healthy' : s.status === 'critical' ? 'critical' : 'warning'}">${s.status.toUpperCase()}</span></td>
                    <td>${s.total_logs.toLocaleString()}</td>
                    <td>${s.errors.toLocaleString()}</td>
                    <td>${s.error_rate}%</td>
                  </tr>
                `).join('')}
              </table>
              
              <footer>
                Report generated by LogMind AI • ${new Date().toISOString()}
              </footer>
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
      <div className="h-full flex items-center justify-center bg-gradient-to-br from-slate-50 to-slate-100">
        <div className="text-center">
          <div className="relative">
            <div className="w-16 h-16 rounded-full border-4 border-blue-200 border-t-blue-500 animate-spin mx-auto" />
            <Database className="absolute inset-0 m-auto text-blue-500" size={24} />
          </div>
          <p className="text-slate-600 font-medium mt-4">Generating report...</p>
          <p className="text-slate-400 text-sm">Analyzing your logs</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="h-full flex items-center justify-center bg-gradient-to-br from-slate-50 to-slate-100">
        <div className="text-center bg-white p-8 rounded-2xl shadow-lg border border-red-100">
          <div className="w-16 h-16 rounded-full bg-red-100 flex items-center justify-center mx-auto mb-4">
            <AlertTriangle size={32} className="text-red-500" />
          </div>
          <p className="text-slate-800 font-semibold text-lg">{error}</p>
          <button
            onClick={loadReport}
            className="mt-6 px-6 py-3 bg-blue-500 text-white rounded-xl hover:bg-blue-600 flex items-center gap-2 mx-auto font-medium"
          >
            <RefreshCw size={18} /> Try Again
          </button>
        </div>
      </div>
    );
  }

  if (!report) return null;

  const { performance: perf, security: sec, services_health: services } = report;
  const maxServiceLogs = Math.max(...services.map(s => s.total_logs), 1);
  const maxCategoryCount = Math.max(...Object.values(sec.categories), 1);

  return (
    <div ref={reportRef} className="h-full overflow-auto bg-gradient-to-br from-slate-50 via-white to-slate-50">
      <div className="max-w-7xl mx-auto p-6 space-y-6">
        {/* Header */}
        <div className="bg-white rounded-2xl p-6 border border-slate-100 shadow-sm">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-blue-500/25">
                <FileText size={28} className="text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-slate-900">Log Analysis Report</h1>
                <p className="text-slate-500 text-sm mt-0.5">
                  {serviceName ? serviceName : 'All Services'} • {timeRange} • 
                  Generated {new Date(report.generated_at).toLocaleString()}
                </p>
              </div>
            </div>
            <div className="flex gap-3">
              <button
                onClick={loadReport}
                className="px-4 py-2.5 bg-slate-100 text-slate-700 rounded-xl hover:bg-slate-200 flex items-center gap-2 font-medium transition-colors"
              >
                <RefreshCw size={18} /> Refresh
              </button>
              <button
                onClick={exportToPDF}
                className="px-4 py-2.5 bg-gradient-to-r from-blue-500 to-indigo-600 text-white rounded-xl hover:from-blue-600 hover:to-indigo-700 flex items-center gap-2 font-medium shadow-lg shadow-blue-500/25 transition-all"
              >
                <Download size={18} /> Export PDF
              </button>
            </div>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="flex gap-2 bg-white rounded-xl p-1.5 border border-slate-100 shadow-sm w-fit">
          {[
            { id: 'overview', label: 'Overview', icon: Layers },
            { id: 'security', label: 'Security', icon: Shield },
            { id: 'services', label: 'Services', icon: Server },
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`px-5 py-2.5 flex items-center gap-2 font-medium rounded-lg transition-all ${
                activeTab === tab.id
                  ? 'bg-gradient-to-r from-blue-500 to-indigo-600 text-white shadow-lg shadow-blue-500/25'
                  : 'text-slate-500 hover:text-slate-700 hover:bg-slate-50'
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
            {/* Main Stats */}
            <div className="grid grid-cols-5 gap-4">
              <StatCard 
                icon={Database} 
                label="Total Logs" 
                value={perf.total_logs}
                subValue={`${perf.template_efficiency.toFixed(0)} logs/template avg`}
                color="blue"
                size="large"
              />
              <StatCard 
                icon={Layers} 
                label="Templates" 
                value={perf.total_templates}
                subValue={`${perf.embedded_templates.toLocaleString()} AI-indexed`}
                color="purple"
              />
              <StatCard 
                icon={Server} 
                label="Services" 
                value={perf.services_count}
                trend="neutral"
                color="slate"
              />
              <StatCard 
                icon={AlertTriangle} 
                label="Error Rate" 
                value={`${perf.error_rate}%`}
                trend={perf.error_rate > 5 ? 'down' : perf.error_rate > 2 ? 'neutral' : 'up'}
                color={perf.error_rate > 5 ? 'red' : perf.error_rate > 2 ? 'amber' : 'green'}
              />
              <StatCard 
                icon={Zap} 
                label="AI Coverage" 
                value={`${perf.embedding_coverage}%`}
                trend={perf.embedding_coverage > 50 ? 'up' : 'neutral'}
                color="green"
              />
            </div>

            {/* Severity Breakdown */}
            <div className="bg-white rounded-2xl p-6 border border-slate-100 shadow-sm">
              <h3 className="text-lg font-bold text-slate-800 mb-4">Severity Breakdown</h3>
              <div className="grid grid-cols-5 gap-4">
                <SeverityBadge level="critical" count={perf.severity_breakdown.critical} />
                <SeverityBadge level="error" count={perf.severity_breakdown.error} />
                <SeverityBadge level="warning" count={perf.severity_breakdown.warning} />
                <SeverityBadge level="info" count={perf.severity_breakdown.info} />
                <SeverityBadge level="debug" count={perf.severity_breakdown.debug} />
              </div>
            </div>

            {/* Hourly Activity */}
            <div className="bg-white rounded-2xl p-6 border border-slate-100 shadow-sm">
              <h3 className="text-lg font-bold text-slate-800 mb-4 flex items-center gap-2">
                <Clock size={20} className="text-blue-500" />
                Hourly Activity (Last 24 Hours)
              </h3>
              <ActivityGrid data={perf.hourly_trend} />
            </div>

            {/* Performance Insights */}
            <div className="grid grid-cols-3 gap-4">
              <div className="bg-gradient-to-br from-blue-500 to-indigo-600 rounded-2xl p-6 text-white">
                <Activity size={24} className="opacity-80 mb-3" />
                <p className="text-4xl font-bold">{perf.template_efficiency.toFixed(0)}</p>
                <p className="text-blue-100 mt-1">Logs per Template</p>
                <p className="text-blue-200 text-xs mt-2">Higher = more pattern reuse</p>
              </div>
              <div className="bg-gradient-to-br from-emerald-500 to-teal-600 rounded-2xl p-6 text-white">
                <TrendingUp size={24} className="opacity-80 mb-3" />
                <p className="text-4xl font-bold">{perf.embedded_templates.toLocaleString()}</p>
                <p className="text-emerald-100 mt-1">AI-Indexed Templates</p>
                <p className="text-emerald-200 text-xs mt-2">Ready for semantic search</p>
              </div>
              <div className="bg-gradient-to-br from-amber-500 to-orange-600 rounded-2xl p-6 text-white">
                <AlertCircle size={24} className="opacity-80 mb-3" />
                <p className="text-4xl font-bold">{perf.warning_rate}%</p>
                <p className="text-amber-100 mt-1">Warning Rate</p>
                <p className="text-amber-200 text-xs mt-2">Non-critical issues</p>
              </div>
            </div>
          </div>
        )}

        {/* Security Tab */}
        {activeTab === 'security' && (
          <div className="space-y-6">
            {/* Security Score Hero */}
            <div className="bg-white rounded-2xl p-8 border border-slate-100 shadow-sm">
              <div className="flex items-center gap-12">
                <div className="text-center">
                  <ProgressRing 
                    value={sec.security_score} 
                    size={160}
                    color={sec.security_score >= 70 ? '#10b981' : sec.security_score >= 50 ? '#f59e0b' : '#ef4444'}
                  />
                  <div className={`mt-4 inline-flex items-center gap-2 px-4 py-2 rounded-full font-semibold ${
                    sec.security_score >= 70 ? 'bg-emerald-100 text-emerald-700' :
                    sec.security_score >= 50 ? 'bg-amber-100 text-amber-700' :
                    'bg-red-100 text-red-700'
                  }`}>
                    <Shield size={18} />
                    {sec.risk_level} Risk
                  </div>
                </div>
                
                <div className="flex-1">
                  <h3 className="text-xl font-bold text-slate-800 mb-2">Security Assessment</h3>
                  <p className="text-slate-500 mb-6">
                    {sec.security_score >= 70 
                      ? 'Your system shows good security posture with minimal threats detected.'
                      : sec.security_score >= 50
                        ? 'Some security concerns detected. Review the recommendations below.'
                        : 'Critical security issues detected. Immediate action recommended.'}
                  </p>
                  
                  <div className="grid grid-cols-3 gap-4">
                    <div className="text-center p-4 bg-slate-50 rounded-xl">
                      <p className="text-2xl font-bold text-slate-800">{sec.total_security_events.toLocaleString()}</p>
                      <p className="text-sm text-slate-500">Security Events</p>
                    </div>
                    <div className="text-center p-4 bg-slate-50 rounded-xl">
                      <p className="text-2xl font-bold text-slate-800">{sec.top_affected_services.length}</p>
                      <p className="text-sm text-slate-500">Affected Services</p>
                    </div>
                    <div className="text-center p-4 bg-slate-50 rounded-xl">
                      <p className="text-2xl font-bold text-slate-800">{sec.recommendations.length}</p>
                      <p className="text-sm text-slate-500">Recommendations</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Threat Categories */}
            <div className="bg-white rounded-2xl p-6 border border-slate-100 shadow-sm">
              <h3 className="text-lg font-bold text-slate-800 mb-4">Security Events by Category</h3>
              <div className="grid grid-cols-2 gap-4">
                {Object.entries(sec.categories).map(([category, count]) => (
                  <ThreatCard 
                    key={category} 
                    category={category} 
                    count={count} 
                    maxCount={maxCategoryCount} 
                  />
                ))}
              </div>
            </div>

            {/* Recommendations */}
            <div className="bg-white rounded-2xl p-6 border border-slate-100 shadow-sm">
              <h3 className="text-lg font-bold text-slate-800 mb-4 flex items-center gap-2">
                <Zap size={20} className="text-amber-500" />
                Recommendations
              </h3>
              <div className="space-y-3">
                {sec.recommendations.map((rec, idx) => (
                  <RecommendationCard key={idx} text={rec} index={idx} />
                ))}
              </div>
            </div>

            {/* Top Affected Services */}
            {sec.top_affected_services.length > 0 && (
              <div className="bg-white rounded-2xl p-6 border border-slate-100 shadow-sm">
                <h3 className="text-lg font-bold text-slate-800 mb-4">Most Affected Services</h3>
                <div className="space-y-3">
                  {sec.top_affected_services.slice(0, 8).map((service, idx) => {
                    const maxErrors = sec.top_affected_services[0]?.error_count || 1;
                    const percentage = (service.error_count / maxErrors) * 100;
                    return (
                      <div key={idx} className="flex items-center gap-4">
                        <div className="w-8 h-8 rounded-lg bg-red-100 flex items-center justify-center text-red-600 font-bold text-sm">
                          {idx + 1}
                        </div>
                        <div className="flex-1">
                          <div className="flex items-center justify-between mb-1">
                            <span className="font-medium text-slate-700">{service.service}</span>
                            <span className="text-red-600 font-semibold">{service.error_count.toLocaleString()} errors</span>
                          </div>
                          <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                            <div 
                              className="h-full bg-gradient-to-r from-red-400 to-red-500 rounded-full"
                              style={{ width: `${percentage}%` }}
                            />
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Services Tab */}
        {activeTab === 'services' && (
          <div className="space-y-6">
            {/* Status Summary */}
            <div className="grid grid-cols-4 gap-4">
              {[
                { status: 'healthy', icon: CheckCircle, bgColor: 'bg-emerald-50', borderColor: 'border-emerald-100', iconBg: 'bg-emerald-100', iconColor: 'text-emerald-600' },
                { status: 'warning', icon: AlertCircle, bgColor: 'bg-amber-50', borderColor: 'border-amber-100', iconBg: 'bg-amber-100', iconColor: 'text-amber-600' },
                { status: 'degraded', icon: AlertTriangle, bgColor: 'bg-orange-50', borderColor: 'border-orange-100', iconBg: 'bg-orange-100', iconColor: 'text-orange-600' },
                { status: 'critical', icon: XCircle, bgColor: 'bg-red-50', borderColor: 'border-red-100', iconBg: 'bg-red-100', iconColor: 'text-red-600' },
              ].map(({ status, icon: Icon, bgColor, borderColor, iconBg, iconColor }) => {
                const count = services.filter(s => s.status === status).length;
                return (
                  <div key={status} className={`${bgColor} rounded-2xl p-5 border ${borderColor}`}>
                    <div className="flex items-center gap-3 mb-3">
                      <div className={`p-2 rounded-xl ${iconBg}`}>
                        <Icon size={20} className={iconColor} />
                      </div>
                      <span className="font-semibold capitalize text-slate-700">{status}</span>
                    </div>
                    <p className="text-4xl font-bold text-slate-800">{count}</p>
                    <p className="text-sm text-slate-500">services</p>
                  </div>
                );
              })}
            </div>

            {/* Services List */}
            <div className="bg-white rounded-2xl p-6 border border-slate-100 shadow-sm">
              <h3 className="text-lg font-bold text-slate-800 mb-4">All Services</h3>
              <div className="grid grid-cols-2 gap-4">
                {services.map((service, idx) => (
                  <ServiceRow key={idx} service={service} maxLogs={maxServiceLogs} />
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
