'use client';

import { useState, useEffect } from 'react';
import { FileText, ChevronRight, Hash, Loader2, AlertCircle } from 'lucide-react';
import { getTopTemplates, getTemplateDetail, getTimeRange } from '@/lib/api';
import type { Template, TemplateDetail } from '@/types';
import { SEVERITY_COLORS } from '@/types';

interface Props {
  serviceName: string;
  timeRange: string;
}

export default function TemplatesView({ serviceName, timeRange }: Props) {
  const [templates, setTemplates] = useState<Template[]>([]);
  const [selectedTemplate, setSelectedTemplate] = useState<TemplateDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [severityMin, setSeverityMin] = useState(0);
  const [selectedHash, setSelectedHash] = useState<string | null>(null);  // String for 64-bit ints

  const loadTemplates = async () => {
    setLoading(true);
    setError(null);

    try {
      const { from, to } = getTimeRange(timeRange);
      const response = await getTopTemplates({
        service_name: serviceName, // Empty string = all services
        from,
        to,
        severity_min: severityMin,
        limit: 50,
      });

      setTemplates(response.templates);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load templates');
    } finally {
      setLoading(false);
    }
  };

  const loadTemplateDetail = async (templateHash: string) => {
    console.log('Loading template detail for hash:', templateHash);
    setSelectedHash(templateHash);
    setDetailLoading(true);
    setError(null);

    try {
      const { from, to } = getTimeRange(timeRange);
      const detail = await getTemplateDetail(templateHash, {
        service_name: serviceName, // Empty string = any service
        from,
        to,
      });

      console.log('Template detail loaded:', detail);
      setSelectedTemplate(detail);
    } catch (e) {
      console.error('Failed to load template detail:', e);
      setError(e instanceof Error ? e.message : 'Failed to load template detail');
    } finally {
      setDetailLoading(false);
    }
  };

  useEffect(() => {
    loadTemplates();
    setSelectedTemplate(null);
    setSelectedHash(null);
  }, [serviceName, timeRange, severityMin]);

  return (
    <div className="h-full flex bg-white">
      {/* Templates list */}
      <div className="w-1/2 border-r border-slate-200 flex flex-col">
        <div className="p-4 border-b border-slate-200 flex items-center justify-between bg-gradient-to-r from-slate-50 to-white">
          <div>
            <h2 className="font-bold text-slate-900 text-lg">Top Templates</h2>
            <p className="text-xs text-slate-500 mt-0.5">
              {serviceName ? `Service: ${serviceName}` : 'All Services'}
            </p>
          </div>
          <select
            value={severityMin}
            onChange={(e) => setSeverityMin(parseInt(e.target.value))}
            className="px-3 py-2 bg-white border border-slate-300 rounded-lg text-sm text-slate-700 font-medium focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 cursor-pointer shadow-sm"
          >
            <option value="0">All Severities</option>
            <option value="2">INFO+</option>
            <option value="3">WARN+</option>
            <option value="4">ERROR+</option>
          </select>
        </div>

        {error && (
          <div className="p-4 bg-red-50 border-b border-red-200 flex items-center gap-2">
            <AlertCircle size={16} className="text-red-500 flex-shrink-0" />
            <p className="text-red-700 text-sm">{error}</p>
          </div>
        )}

        <div className="flex-1 overflow-auto bg-slate-50/50">
          {loading ? (
            <div className="p-12 text-center">
              <Loader2 size={32} className="animate-spin mx-auto text-blue-500 mb-3" />
              <p className="text-slate-600 font-medium">Loading templates...</p>
            </div>
          ) : templates.length === 0 ? (
            <div className="p-12 text-center">
              <FileText size={48} className="mx-auto text-slate-300 mb-3" />
              <p className="text-slate-600 font-medium">No templates found</p>
              <p className="text-slate-400 text-sm mt-1">Try adjusting the time range or service filter</p>
            </div>
          ) : (
            <ul className="divide-y divide-slate-200">
              {templates.map((template) => (
                <li
                  key={template.template_hash}
                  className={`p-4 cursor-pointer flex items-start gap-3 transition-all duration-150 ${
                    selectedHash === template.template_hash
                      ? 'bg-blue-50 border-l-4 border-l-blue-500 shadow-sm'
                      : 'bg-white hover:bg-slate-50 border-l-4 border-l-transparent'
                  }`}
                  onClick={() => loadTemplateDetail(template.template_hash)}
                >
                  <FileText size={18} className={`mt-0.5 flex-shrink-0 ${
                    selectedHash === template.template_hash ? 'text-blue-500' : 'text-slate-400'
                  }`} />
                  <div className="flex-1 min-w-0">
                    <p className={`text-sm font-mono truncate ${
                      selectedHash === template.template_hash ? 'text-blue-900' : 'text-slate-800'
                    }`}>
                      {template.template_text}
                    </p>
                    <div className="mt-2 flex items-center gap-4 text-xs">
                      <span className="font-bold text-blue-600 bg-blue-50 px-2 py-0.5 rounded">
                        {template.count.toLocaleString()} hits
                      </span>
                      <span className="text-slate-400 flex items-center gap-1 font-mono">
                        <Hash size={12} /> {template.template_hash}
                      </span>
                    </div>
                  </div>
                  <ChevronRight size={18} className={`mt-0.5 transition-colors ${
                    selectedHash === template.template_hash ? 'text-blue-500' : 'text-slate-300'
                  }`} />
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>

      {/* Template detail */}
      <div className="w-1/2 flex flex-col bg-slate-50">
        {detailLoading ? (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center">
              <Loader2 size={32} className="animate-spin mx-auto text-blue-500 mb-3" />
              <p className="text-slate-600 font-medium">Loading template details...</p>
            </div>
          </div>
        ) : selectedTemplate ? (
          <>
            <div className="p-4 border-b border-slate-200 bg-white">
              <h2 className="font-bold text-slate-900 text-lg">Template Detail</h2>
              <p className="text-xs text-slate-500 mt-1 font-mono bg-slate-100 inline-block px-2 py-0.5 rounded">
                #{selectedTemplate.template_hash}
              </p>
            </div>

            <div className="flex-1 overflow-auto p-4 space-y-5">
              {/* Template text */}
              <div className="bg-white rounded-xl border border-slate-200 overflow-hidden shadow-sm">
                <div className="px-4 py-2 bg-slate-100 border-b border-slate-200">
                  <h3 className="text-sm font-bold text-slate-700">Template Pattern</h3>
                </div>
                <pre className="p-4 text-sm font-mono overflow-x-auto whitespace-pre-wrap text-slate-800 bg-white">
                  {selectedTemplate.template_text}
                </pre>
              </div>

              {/* Stats */}
              <div className="grid grid-cols-2 gap-4">
                <div className="p-4 bg-gradient-to-br from-blue-500 to-blue-600 rounded-xl shadow-lg shadow-blue-500/20">
                  <p className="text-3xl font-bold text-white">
                    {selectedTemplate.total_count.toLocaleString()}
                  </p>
                  <p className="text-sm text-blue-100 font-medium mt-1">Total Occurrences</p>
                </div>
                <div className="p-4 bg-white rounded-xl border border-slate-200 shadow-sm">
                  <p className={`text-sm font-bold ${
                    selectedTemplate.embedding_state === 'ready' ? 'text-green-600' : 'text-slate-500'
                  }`}>
                    {selectedTemplate.embedding_state === 'ready' ? '✓ AI Ready' : '○ Pending Training'}
                  </p>
                  <p className="text-xs text-slate-500 mt-2">
                    First seen: {new Date(selectedTemplate.first_seen_utc).toLocaleDateString()}
                  </p>
                  <p className="text-xs text-slate-500">
                    Service: {selectedTemplate.service_name || 'Unknown'}
                  </p>
                </div>
              </div>

              {/* Recent occurrences */}
              <div className="bg-white rounded-xl border border-slate-200 overflow-hidden shadow-sm">
                <div className="px-4 py-2 bg-slate-100 border-b border-slate-200">
                  <h3 className="text-sm font-bold text-slate-700">Recent Occurrences</h3>
                </div>
                <div className="divide-y divide-slate-100">
                  {selectedTemplate.occurrences.map((occ) => (
                    <div key={occ.log_id} className="p-4">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-slate-800 font-medium text-sm">
                          {new Date(occ.timestamp_utc).toLocaleString()}
                        </span>
                        <span className={`text-xs font-bold px-2 py-0.5 rounded ${SEVERITY_COLORS[occ.severity]} bg-opacity-10`}>
                          Level {occ.severity}
                        </span>
                      </div>
                      {occ.parameters.length > 0 && (
                        <div className="flex flex-wrap gap-1.5">
                          {occ.parameters.map((param, i) => (
                            <span
                              key={i}
                              className="px-2 py-1 bg-amber-50 text-amber-700 border border-amber-200 rounded-lg text-xs font-mono"
                            >
                              {param}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center bg-gradient-to-br from-slate-50 to-slate-100">
            <div className="text-center p-8">
              <div className="w-20 h-20 mx-auto mb-4 bg-slate-200 rounded-2xl flex items-center justify-center">
                <ChevronRight size={40} className="text-slate-400" />
              </div>
              <p className="text-xl font-bold text-slate-700">Select a Template</p>
              <p className="text-slate-500 mt-2 max-w-xs">
                Click on any template from the list to view its details and recent occurrences
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
