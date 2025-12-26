'use client';

import { useState, useEffect, useCallback } from 'react';
import { Search, ChevronDown, ChevronRight, RefreshCw, ToggleLeft, ToggleRight } from 'lucide-react';
import { queryLogs, getTimeRange, getQuickStats } from '@/lib/api';
import type { LogEvent, LogQueryResponse } from '@/types';
import { SEVERITY_COLORS, SEVERITY_BG_COLORS, SEVERITY_NAMES } from '@/types';

interface Props {
  serviceName: string;
  timeRange: string;
  refreshTrigger?: number;
}

export default function LogExplorer({ serviceName, timeRange, refreshTrigger }: Props) {
  const [logs, setLogs] = useState<LogEvent[]>([]);
  const [total, setTotal] = useState(0);
  const [estimatedTotal, setEstimatedTotal] = useState<number | null>(null);
  const [hasMore, setHasMore] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [severityFilter, setSeverityFilter] = useState<number | undefined>(undefined);
  const [expandedLogs, setExpandedLogs] = useState<Set<number>>(new Set());
  const [offset, setOffset] = useState(0);
  const [showRawLogs, setShowRawLogs] = useState(true); // Show raw logs by default
  const limit = 100;

  // Fetch estimated total on mount (only once)
  useEffect(() => {
    getQuickStats()
      .then(stats => setEstimatedTotal(stats.logs_estimated))
      .catch(() => {});  // Ignore errors - this is just for display
  }, []);

  // Reset offset when filters change
  useEffect(() => {
    setOffset(0);
  }, [serviceName, timeRange, severityFilter]);

  const loadLogs = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const { from, to } = getTimeRange(timeRange);
      console.log('Querying logs:', { from, to, serviceName, severityFilter, limit, offset });
      const response = await queryLogs({
        from,
        to,
        service_name: serviceName || undefined,
        severity_min: severityFilter,
        limit,
        offset,
      });
      console.log('Logs response:', response);

      setLogs(response.logs);
      setTotal(response.total);
      setHasMore(response.has_more);
    } catch (e) {
      console.error('Load logs error:', e);
      setError(e instanceof Error ? e.message : 'Failed to load logs');
    } finally {
      setLoading(false);
    }
  }, [serviceName, timeRange, severityFilter, offset]);

  useEffect(() => {
    loadLogs();
  }, [loadLogs, refreshTrigger]);

  const toggleExpand = (logId: number) => {
    const newExpanded = new Set(expandedLogs);
    if (newExpanded.has(logId)) {
      newExpanded.delete(logId);
    } else {
      newExpanded.add(logId);
    }
    setExpandedLogs(newExpanded);
  };

  const formatTimestamp = (ts: string) => {
    try {
      return new Date(ts).toLocaleString();
    } catch {
      return ts;
    }
  };

  const filteredLogs = logs.filter((log) => {
    if (!searchQuery) return true;
    const query = searchQuery.toLowerCase();
    return (
      log.body_raw.toLowerCase().includes(query) ||
      log.template_text?.toLowerCase().includes(query) ||
      log.service_name.toLowerCase().includes(query)
    );
  });

  return (
    <div className="h-full flex flex-col bg-gradient-to-br from-slate-50 via-white to-slate-50">
      {/* Header with Filters */}
      <div className="p-5 border-b border-slate-200/80 flex items-center gap-4 bg-white/80 backdrop-blur-xl shadow-sm">
        <div className="relative flex-1 max-w-lg">
          <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400" size={18} />
          <input
            type="text"
            placeholder="Search in logs..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-12 pr-4 py-3 border-2 border-slate-200 rounded-2xl text-sm focus:outline-none focus:ring-4 focus:ring-blue-500/20 focus:border-blue-400 bg-white transition-all shadow-sm placeholder:text-slate-400"
          />
        </div>

        <select
          value={severityFilter ?? ''}
          onChange={(e) => setSeverityFilter(e.target.value ? parseInt(e.target.value) : undefined)}
          className="px-4 py-3 border-2 border-slate-200 rounded-2xl text-sm focus:outline-none focus:ring-4 focus:ring-blue-500/20 focus:border-blue-400 bg-white transition-all cursor-pointer font-medium shadow-sm"
        >
          <option value="">All Severities</option>
          <option value="3">WARN+</option>
          <option value="4">ERROR+</option>
          <option value="5">FATAL only</option>
        </select>

        <button
          onClick={() => loadLogs()}
          disabled={loading}
          className="flex items-center gap-2 px-3 py-2 text-slate-600 hover:bg-slate-100 rounded-xl transition-all disabled:opacity-50"
          title="Refresh logs"
        >
          <RefreshCw size={16} className={loading ? 'animate-spin' : ''} />
          Refresh
        </button>

        <button
          onClick={() => setShowRawLogs(!showRawLogs)}
          className={`flex items-center gap-2 px-3 py-2 rounded-xl transition-all ${
            showRawLogs 
              ? 'bg-blue-100 text-blue-700' 
              : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
          }`}
          title={showRawLogs ? "Showing raw logs" : "Showing templates"}
        >
          {showRawLogs ? <ToggleRight size={16} /> : <ToggleLeft size={16} />}
          {showRawLogs ? 'Raw' : 'Templates'}
        </button>

        <span className="text-sm text-slate-500 font-medium">
          {hasMore && estimatedTotal ? (
            <>~{estimatedTotal.toLocaleString()} logs</>
          ) : (
            <>{total.toLocaleString()} logs found</>
          )}
        </span>
      </div>

      {/* Error */}
      {error && (
        <div className="p-4 bg-rose-50 text-rose-700 text-sm border-b border-rose-200 flex items-center justify-between">
          <span>{error}</span>
          <span className="text-xs text-rose-500">
            💡 Tip: Select a specific service or shorter time range for faster results
          </span>
        </div>
      )}

      {/* Logs table */}
      <div className="flex-1 overflow-auto px-4 py-3">
        {loading ? (
          <div className="flex flex-col items-center justify-center h-64 text-slate-500">
            <div className="w-16 h-16 border-4 border-blue-200 border-t-blue-500 rounded-full animate-spin mb-4"></div>
            <p className="font-semibold text-lg">Loading logs...</p>
            <p className="text-sm text-slate-400 mt-1">
              Fetching your log data
            </p>
          </div>
        ) : filteredLogs.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-64 text-slate-500">
            <div className="w-20 h-20 bg-slate-100 rounded-full flex items-center justify-center mb-4">
              <Search size={32} className="text-slate-300" />
            </div>
            <p className="font-semibold text-lg">No logs found</p>
            <p className="text-sm text-slate-400 mt-1">Try adjusting filters or ingest some logs</p>
          </div>
        ) : (
          <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-gradient-to-r from-slate-50 to-slate-100 sticky top-0 border-b border-slate-200">
              <tr>
                <th className="w-10"></th>
                <th className="px-4 py-4 text-left font-bold text-slate-700 uppercase text-xs tracking-wider">Timestamp</th>
                <th className="px-4 py-4 text-left font-bold text-slate-700 uppercase text-xs tracking-wider w-24">Level</th>
                <th className="px-4 py-4 text-left font-bold text-slate-700 uppercase text-xs tracking-wider">Service</th>
                <th className="px-4 py-4 text-left font-bold text-slate-700 uppercase text-xs tracking-wider">Message</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {filteredLogs.map((log) => (
                <>
                  <tr
                    key={log.id}
                    className={`hover:bg-blue-50/50 cursor-pointer transition-all duration-150 ${
                      expandedLogs.has(log.id) ? 'bg-blue-50/30' : ''
                    }`}
                    onClick={() => toggleExpand(log.id)}
                  >
                    <td className="px-3 py-3">
                      <div className={`w-6 h-6 rounded-lg flex items-center justify-center transition-all ${
                        expandedLogs.has(log.id) ? 'bg-blue-100 text-blue-600' : 'bg-slate-100 text-slate-400'
                      }`}>
                        {expandedLogs.has(log.id) ? (
                          <ChevronDown size={14} />
                        ) : (
                          <ChevronRight size={14} />
                        )}
                      </div>
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap">
                      <span className="text-slate-500 font-mono text-xs bg-slate-50 px-2 py-1 rounded-lg">
                        {formatTimestamp(log.timestamp_utc)}
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      <span className={`inline-flex items-center px-2.5 py-1 rounded-full text-xs font-bold uppercase tracking-wider ${
                        log.severity >= 4 ? 'bg-red-100 text-red-700' :
                        log.severity === 3 ? 'bg-amber-100 text-amber-700' :
                        log.severity === 2 ? 'bg-blue-100 text-blue-700' :
                        'bg-slate-100 text-slate-600'
                      }`}>
                        {log.severity_name}
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      <span className="inline-flex items-center px-2.5 py-1 bg-indigo-50 text-indigo-700 rounded-lg text-xs font-medium">
                        {log.service_name}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-slate-800 max-w-2xl">
                      <div className="break-words whitespace-pre-wrap font-mono text-xs leading-relaxed">
                        {showRawLogs ? (log.body_raw || log.template_text) : (log.template_text || log.body_raw)}
                      </div>
                    </td>
                  </tr>
                  {expandedLogs.has(log.id) && (
                    <tr key={`${log.id}-expanded`} className="bg-gradient-to-r from-slate-50 via-blue-50/20 to-indigo-50/20">
                      <td colSpan={5} className="px-6 py-5">
                        <div className="space-y-5">
                          <div>
                            <span className="inline-flex items-center gap-2 text-xs font-bold text-slate-600 uppercase tracking-wider mb-3">
                              <div className="w-1 h-4 bg-gradient-to-b from-blue-500 to-indigo-500 rounded-full"></div>
                              Raw Message
                            </span>
                            <pre className="p-4 bg-gradient-to-br from-slate-900 to-slate-800 text-green-400 border border-slate-700 rounded-2xl text-xs overflow-x-auto font-mono shadow-lg leading-relaxed">
{log.body_raw}
                            </pre>
                          </div>
                          {log.template_text && (
                            <div>
                              <span className="inline-flex items-center gap-2 text-xs font-bold text-slate-600 uppercase tracking-wider mb-3">
                                <div className="w-1 h-4 bg-gradient-to-b from-sky-500 to-cyan-500 rounded-full"></div>
                                Template Pattern
                              </span>
                              <pre className="p-4 bg-gradient-to-br from-sky-50 to-cyan-50 border-2 border-sky-200 rounded-2xl text-xs font-mono text-sky-800 shadow-sm leading-relaxed">
{log.template_text}
                              </pre>
                            </div>
                          )}
                          {log.parameters.length > 0 && (
                            <div>
                              <span className="inline-flex items-center gap-2 text-xs font-bold text-slate-600 uppercase tracking-wider mb-3">
                                <div className="w-1 h-4 bg-gradient-to-b from-amber-500 to-orange-500 rounded-full"></div>
                                Extracted Values
                              </span>
                              <div className="flex flex-wrap gap-2">
                                {log.parameters.map((param, i) => (
                                  <span
                                    key={i}
                                    className="px-3 py-1.5 bg-gradient-to-r from-amber-50 to-orange-50 text-amber-800 rounded-xl text-xs font-medium border border-amber-200 shadow-sm"
                                  >
                                    {String(param)}
                                  </span>
                                ))}
                              </div>
                            </div>
                          )}
                          <div className="flex gap-6 text-xs text-slate-500 pt-2 border-t border-slate-200">
                            <span className="font-mono bg-slate-100 px-2 py-1 rounded-lg">ID: {log.id}</span>
                            <span className="font-mono bg-slate-100 px-2 py-1 rounded-lg">Template: {String(log.template_hash).slice(0, 12)}...</span>
                            {log.host && <span className="bg-slate-100 px-2 py-1 rounded-lg">Host: {log.host}</span>}
                            {log.trace_id && <span className="bg-slate-100 px-2 py-1 rounded-lg">Trace: {log.trace_id}</span>}
                          </div>
                        </div>
                      </td>
                    </tr>
                  )}
                </>
              ))}
            </tbody>
          </table>
          </div>
        )}
      </div>

      {/* Pagination */}
      {logs.length > 0 && (
        <div className="px-5 py-4 border-t border-slate-200/80 flex items-center justify-between bg-white/80 backdrop-blur-xl">
          <span className="text-sm text-slate-600 font-medium">
            Showing <span className="font-bold text-slate-800">{offset + 1}-{offset + logs.length}</span>
            {hasMore ? (
              <span className="text-emerald-600 ml-1">• More available</span>
            ) : (
              <span className="text-slate-400"> of {total.toLocaleString()}</span>
            )}
          </span>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setOffset(0)}
              disabled={offset === 0}
              className="px-4 py-2 border-2 border-slate-200 rounded-xl text-sm disabled:opacity-40 hover:bg-slate-50 hover:border-slate-300 transition-all font-medium shadow-sm"
            >
              ⟨⟨ First
            </button>
            <button
              onClick={() => setOffset(Math.max(0, offset - limit))}
              disabled={offset === 0}
              className="px-4 py-2 border-2 border-slate-200 rounded-xl text-sm disabled:opacity-40 hover:bg-slate-50 hover:border-slate-300 transition-all font-medium shadow-sm"
            >
              ⟨ Prev
            </button>
            
            {/* Current page indicator */}
            <div className="px-5 py-2 bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-500 text-white rounded-xl text-sm font-bold shadow-lg shadow-blue-500/30">
              Page {Math.floor(offset / limit) + 1}
            </div>
            
            <button
              onClick={() => setOffset(offset + limit)}
              disabled={!hasMore}
              className="px-4 py-2 border-2 border-slate-200 rounded-xl text-sm disabled:opacity-40 hover:bg-slate-50 hover:border-slate-300 transition-all font-medium shadow-sm"
            >
              Next ⟩
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
