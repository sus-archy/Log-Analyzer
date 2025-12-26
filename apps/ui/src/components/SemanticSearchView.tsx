'use client';

import { useState, useEffect } from 'react';
import { 
  Search, Sparkles, Clock, FileText, Hash, Loader2, 
  RefreshCw, AlertCircle, Lightbulb, ArrowRight, Wand2,
  MessageSquare, Target, Zap, Eye
} from 'lucide-react';
import { semanticSearch, getTimeRange, getEmbeddingStats } from '@/lib/api';
import type { TemplateDetail } from '@/types';

interface SearchResult {
  template_hash: string;  // String for 64-bit ints
  template_text: string;
  count: number;
  similarity: number;
  sample_logs: Array<{
    log_id: number;
    timestamp_utc: string;
    content: string;
    severity: number;
  }>;
}

interface Props {
  serviceName: string;
  timeRange: string;
}

const SAMPLE_QUERIES = [
  "authentication failures",
  "database connection errors",
  "timeout exceptions",
  "memory warnings",
  "user login issues",
  "API rate limits",
  "permission denied",
  "network connectivity problems",
];

export default function SemanticSearchView({ serviceName, timeRange }: Props) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searchTime, setSearchTime] = useState<number | null>(null);
  const [embeddingReady, setEmbeddingReady] = useState(false);
  const [selectedResult, setSelectedResult] = useState<SearchResult | null>(null);

  useEffect(() => {
    // Check if embeddings are ready
    const checkEmbeddings = async () => {
      try {
        const stats = await getEmbeddingStats();
        setEmbeddingReady(stats.embedded_count > 0);
      } catch {
        setEmbeddingReady(false);
      }
    };
    checkEmbeddings();
  }, []);

  const handleSearch = async (searchQuery: string = query) => {
    if (!searchQuery.trim()) return;
    
    setLoading(true);
    setError(null);
    setResults([]);
    setSelectedResult(null);

    const startTime = performance.now();

    try {
      const { from, to } = getTimeRange(timeRange);
      const response = await semanticSearch({
        q: searchQuery,
        service_name: serviceName,
        from,
        to,
        limit: 10,
      });

      setSearchTime(performance.now() - startTime);
      
      // Transform response to our format
      const transformedResults: SearchResult[] = response.results.map((r: any) => ({
        template_hash: r.template_hash,
        template_text: r.template_text,
        count: r.count || 0,
        similarity: r.similarity || r.score || 0.85,
        sample_logs: r.sample_logs || [],
      }));

      setResults(transformedResults);
      if (transformedResults.length > 0) {
        setSelectedResult(transformedResults[0]);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Search failed');
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  const getSimilarityColor = (similarity: number) => {
    if (similarity >= 0.9) return 'text-emerald-500 bg-emerald-50';
    if (similarity >= 0.7) return 'text-blue-500 bg-blue-50';
    if (similarity >= 0.5) return 'text-amber-500 bg-amber-50';
    return 'text-slate-500 bg-slate-50';
  };

  return (
    <div className="h-full flex bg-white">
      {/* Search Panel */}
      <div className="w-1/2 border-r border-slate-200 flex flex-col">
        {/* Search Header */}
        <div className="p-6 bg-gradient-to-r from-purple-500 via-fuchsia-500 to-pink-500">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-12 h-12 bg-white/20 backdrop-blur rounded-xl flex items-center justify-center">
              <Sparkles size={24} className="text-white" />
            </div>
            <div>
              <h2 className="font-bold text-white text-lg">AI Semantic Search</h2>
              <p className="text-white/70 text-sm">Natural language log analysis</p>
            </div>
          </div>

          <div className="relative">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Describe what you're looking for..."
              className="w-full px-4 py-3 pl-12 pr-24 bg-white/10 backdrop-blur border border-white/20 rounded-xl text-white placeholder-white/50 focus:outline-none focus:ring-2 focus:ring-white/50"
            />
            <Search size={20} className="absolute left-4 top-1/2 -translate-y-1/2 text-white/50" />
            <button
              onClick={() => handleSearch()}
              disabled={loading || !query.trim()}
              className="absolute right-2 top-1/2 -translate-y-1/2 px-4 py-1.5 bg-white text-purple-600 rounded-lg font-medium text-sm hover:bg-white/90 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
            >
              {loading ? <Loader2 size={16} className="animate-spin" /> : <Wand2 size={16} />}
              Search
            </button>
          </div>

          {!embeddingReady && (
            <div className="mt-3 flex items-center gap-2 text-white/80 text-sm bg-white/10 rounded-lg px-3 py-2">
              <AlertCircle size={16} />
              <span>Train AI first for better results (AI Dashboard → Start Training)</span>
            </div>
          )}
        </div>

        {/* Sample Queries */}
        {results.length === 0 && !loading && (
          <div className="p-4 border-b border-slate-200 bg-slate-50">
            <div className="flex items-center gap-2 text-slate-600 text-sm mb-3">
              <Lightbulb size={16} className="text-amber-500" />
              <span>Try these searches:</span>
            </div>
            <div className="flex flex-wrap gap-2">
              {SAMPLE_QUERIES.map((sq) => (
                <button
                  key={sq}
                  onClick={() => {
                    setQuery(sq);
                    handleSearch(sq);
                  }}
                  className="px-3 py-1.5 bg-white border border-slate-200 rounded-full text-sm text-slate-600 hover:border-purple-300 hover:text-purple-600 transition-colors"
                >
                  {sq}
                </button>
              ))}
            </div>
          </div>
        )}

        {error && (
          <div className="p-4 bg-red-50 border-b border-red-200 flex items-center gap-2 text-red-700">
            <AlertCircle size={16} />
            <span className="text-sm">{error}</span>
          </div>
        )}

        {/* Results */}
        <div className="flex-1 overflow-auto">
          {loading ? (
            <div className="flex items-center justify-center h-64">
              <div className="text-center">
                <div className="relative">
                  <Sparkles size={48} className="text-purple-400 mx-auto animate-pulse" />
                  <div className="absolute inset-0 animate-ping">
                    <Sparkles size={48} className="text-purple-200 mx-auto" />
                  </div>
                </div>
                <p className="text-slate-500 mt-4">Searching through log patterns...</p>
              </div>
            </div>
          ) : results.length > 0 ? (
            <>
              <div className="p-3 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
                <span className="text-sm text-slate-600">
                  Found {results.length} relevant patterns
                </span>
                {searchTime && (
                  <span className="text-xs text-slate-400 flex items-center gap-1">
                    <Zap size={12} />
                    {searchTime.toFixed(0)}ms
                  </span>
                )}
              </div>
              {results.map((result, idx) => (
                <button
                  key={result.template_hash}
                  onClick={() => setSelectedResult(result)}
                  className={`w-full p-4 border-b border-slate-100 hover:bg-slate-50 text-left transition-all ${
                    selectedResult?.template_hash === result.template_hash 
                      ? 'bg-purple-50 border-l-4 border-l-purple-500' 
                      : ''
                  }`}
                >
                  <div className="flex items-start gap-3">
                    <span className="flex-shrink-0 w-6 h-6 bg-purple-100 text-purple-600 rounded-full flex items-center justify-center text-xs font-bold">
                      {idx + 1}
                    </span>
                    <div className="flex-1 min-w-0">
                      <p className="font-mono text-sm text-slate-800 truncate">
                        {result.template_text}
                      </p>
                      <div className="flex items-center gap-4 mt-2">
                        <span className={`px-2 py-0.5 rounded text-xs font-medium ${getSimilarityColor(result.similarity)}`}>
                          {(result.similarity * 100).toFixed(0)}% match
                        </span>
                        <span className="text-xs text-slate-400 flex items-center gap-1">
                          <Eye size={12} />
                          {result.count.toLocaleString()} logs
                        </span>
                      </div>
                    </div>
                    <ArrowRight size={16} className="text-slate-400 flex-shrink-0" />
                  </div>
                </button>
              ))}
            </>
          ) : null}
        </div>
      </div>

      {/* Detail Panel */}
      <div className="w-1/2 flex flex-col bg-slate-50">
        {selectedResult ? (
          <>
            <div className="p-6 bg-white border-b border-slate-200">
              <div className="flex items-center gap-2 mb-3">
                <span className={`px-3 py-1 rounded-lg text-sm font-medium ${getSimilarityColor(selectedResult.similarity)}`}>
                  {(selectedResult.similarity * 100).toFixed(0)}% Semantic Match
                </span>
              </div>
              <div className="p-4 bg-slate-900 rounded-xl">
                <code className="text-emerald-400 text-sm font-mono break-all">
                  {selectedResult.template_text}
                </code>
              </div>
              <div className="flex items-center gap-4 mt-4 text-sm text-slate-500">
                <span className="flex items-center gap-1">
                  <Hash size={14} />
                  {selectedResult.template_hash}
                </span>
                <span className="flex items-center gap-1">
                  <FileText size={14} />
                  {selectedResult.count.toLocaleString()} occurrences
                </span>
              </div>
            </div>

            <div className="flex-1 overflow-auto p-6">
              <h3 className="font-semibold text-slate-900 mb-4 flex items-center gap-2">
                <MessageSquare size={18} className="text-blue-500" />
                Sample Logs
              </h3>
              
              {selectedResult.sample_logs && selectedResult.sample_logs.length > 0 ? (
                <div className="space-y-3">
                  {selectedResult.sample_logs.slice(0, 5).map((log, idx) => (
                    <div key={log.log_id || idx} className="bg-white rounded-lg p-4 border border-slate-200">
                      <div className="flex items-center gap-2 mb-2 text-xs text-slate-400">
                        <Clock size={12} />
                        {new Date(log.timestamp_utc).toLocaleString()}
                      </div>
                      <p className="text-sm text-slate-700 font-mono">{log.content}</p>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-slate-500 text-sm">No sample logs available</p>
              )}

              {/* AI Insights */}
              <div className="mt-6 p-4 bg-gradient-to-r from-purple-50 to-fuchsia-50 rounded-xl border border-purple-100">
                <div className="flex items-center gap-2 mb-3">
                  <Sparkles size={16} className="text-purple-500" />
                  <span className="font-medium text-purple-900">AI Insights</span>
                </div>
                <ul className="space-y-2 text-sm text-purple-800">
                  <li className="flex items-start gap-2">
                    <Target size={14} className="text-purple-500 mt-0.5 flex-shrink-0" />
                    This pattern was matched based on semantic similarity to your query.
                  </li>
                  <li className="flex items-start gap-2">
                    <Target size={14} className="text-purple-500 mt-0.5 flex-shrink-0" />
                    The AI understands intent, not just keywords - try different phrasings!
                  </li>
                </ul>
              </div>
            </div>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center px-8">
              <div className="w-20 h-20 bg-gradient-to-br from-purple-100 to-fuchsia-100 rounded-2xl flex items-center justify-center mx-auto mb-4">
                <Search size={32} className="text-purple-400" />
              </div>
              <h3 className="text-lg font-semibold text-slate-800 mb-2">Semantic Search</h3>
              <p className="text-slate-500 text-sm max-w-sm">
                Describe what you're looking for in natural language. 
                Our AI will find the most relevant log patterns.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
