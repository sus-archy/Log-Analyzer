'use client';

import { useState, useEffect, useRef } from 'react';
import { 
  Search, FileText, Database, RefreshCw, 
  AlertCircle, Activity, Clock, Server, Zap, ChevronRight,
  BarChart2, TrendingUp, Settings, Upload, FolderOpen, Brain,
  Sparkles, Shield, Cpu, MessageCircle
} from 'lucide-react';
import { getServices, ingestFromFolder, processEmbeddings, healthCheck, uploadLogFile, getEmbeddingStats, type EmbeddingStats } from '@/lib/api';
import LogExplorer from '@/components/LogExplorer';
import TemplatesView from '@/components/TemplatesView';
import StatsView from '@/components/StatsView';
import FileBrowser from '@/components/FileBrowser';
import AIModelView from '@/components/AIModelView';
import AnomalyView from '@/components/AnomalyView';
import SemanticSearchView from '@/components/SemanticSearchView';
import ServiceSelector from '@/components/ServiceSelector';
import ReportsView from '@/components/ReportsView';
import AIChat from '@/components/AIChat';

type Tab = 'explorer' | 'templates' | 'stats' | 'ai-model' | 'anomaly' | 'semantic' | 'reports' | 'ai-chat';

interface UploadProgress {
  fileName: string;
  progress: number;
  status: 'uploading' | 'processing' | 'done' | 'error';
}

export default function Home() {
  const [activeTab, setActiveTab] = useState<Tab>('explorer');
  const [services, setServices] = useState<string[]>([]);
  const [selectedService, setSelectedService] = useState<string>('');
  const [timeRange, setTimeRange] = useState('all');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [apiStatus, setApiStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [stats, setStats] = useState({ services: 0, logs: 0, templates: 0 });
  const [fileBrowserOpen, setFileBrowserOpen] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<UploadProgress[]>([]);
  const [embeddingStats, setEmbeddingStats] = useState<EmbeddingStats | null>(null);
  const [isTraining, setIsTraining] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const checkApi = async () => {
      try {
        await healthCheck();
        setApiStatus('online');
      } catch (e) {
        setApiStatus('offline');
      }
    };
    checkApi();
    const interval = setInterval(checkApi, 30000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const loadServices = async () => {
      try {
        const data = await getServices();
        setServices(data.services);
        setStats(prev => ({ ...prev, services: data.services.length }));
        if (data.services.length > 0 && !selectedService) {
          setSelectedService(data.services[0]);
        }
      } catch (e) {
        console.error('Failed to load services:', e);
      }
    };
    if (apiStatus === 'online') {
      loadServices();
    }
  }, [apiStatus]);

  // Load embedding stats
  useEffect(() => {
    const loadEmbeddingStats = async () => {
      if (apiStatus !== 'online') return;
      try {
        const stats = await getEmbeddingStats();
        setEmbeddingStats(stats);
      } catch (e) {
        console.error('Failed to load embedding stats:', e);
      }
    };
    loadEmbeddingStats();
    const interval = setInterval(loadEmbeddingStats, 5000); // Refresh every 5 seconds
    return () => clearInterval(interval);
  }, [apiStatus]);

  const handleIngest = async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await ingestFromFolder();
      setStats(prev => ({ 
        ...prev, 
        logs: prev.logs + result.stats.events_inserted,
        templates: prev.templates + result.stats.templates_discovered 
      }));
      const data = await getServices();
      setServices(data.services);
      setStats(prev => ({ ...prev, services: data.services.length }));
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Ingestion failed');
    } finally {
      setLoading(false);
    }
  };

  const handleEmbed = async () => {
    if (isTraining) return;
    setIsTraining(true);
    setError(null);
    
    try {
      // Process embeddings in batches until all are done
      let totalProcessed = 0;
      let processed = 0;
      let consecutiveErrors = 0;
      
      do {
        try {
          processed = (await processEmbeddings(100)).processed;
          totalProcessed += processed;
          consecutiveErrors = 0; // Reset on success
          
          // Refresh stats after each batch
          const stats = await getEmbeddingStats();
          setEmbeddingStats(stats);
          
          // Small delay between batches to avoid overwhelming the server
          if (processed > 0) {
            await new Promise(resolve => setTimeout(resolve, 1000));
          }
        } catch (batchError) {
          consecutiveErrors++;
          console.error('Batch error:', batchError);
          
          // If we get 3 consecutive errors, stop
          if (consecutiveErrors >= 3) {
            throw new Error('Training stopped after multiple errors. Please try again.');
          }
          
          // Wait longer before retry
          await new Promise(resolve => setTimeout(resolve, 3000));
          processed = 1; // Keep looping to retry
        }
      } while (processed > 0 && embeddingStats && embeddingStats.pending_count > 0);
      
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Embedding failed');
    } finally {
      setIsTraining(false);
      // Final stats refresh
      try {
        const stats = await getEmbeddingStats();
        setEmbeddingStats(stats);
      } catch (e) {
        console.error('Failed to get final stats:', e);
      }
    }
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    setError(null);
    const fileArray = Array.from(files);
    
    // Initialize progress for all files
    setUploadProgress(fileArray.map(f => ({
      fileName: f.name,
      progress: 0,
      status: 'uploading' as const,
    })));
    
    for (let i = 0; i < fileArray.length; i++) {
      const file = fileArray[i];
      
      try {
        const result = await uploadLogFile(file, undefined, (progress) => {
          setUploadProgress(prev => prev.map((p, idx) => 
            idx === i ? { ...p, progress, status: progress === 100 ? 'processing' : 'uploading' } : p
          ));
        });
        
        setUploadProgress(prev => prev.map((p, idx) => 
          idx === i ? { ...p, progress: 100, status: 'done' } : p
        ));
        
        setStats(prev => ({
          ...prev,
          logs: prev.logs + result.stats.events_inserted,
          templates: prev.templates + result.stats.templates_discovered,
        }));
      } catch (e) {
        setUploadProgress(prev => prev.map((p, idx) => 
          idx === i ? { ...p, status: 'error' } : p
        ));
        setError(e instanceof Error ? e.message : 'Upload failed');
      }
    }
    
    // Refresh services
    try {
      const data = await getServices();
      setServices(data.services);
      setStats(prev => ({ ...prev, services: data.services.length }));
    } catch (e) {
      console.error('Failed to refresh services:', e);
    }
    
    // Clear progress after delay
    setTimeout(() => {
      setUploadProgress([]);
    }, 3000);
    
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleFileBrowserIngest = (ingestStats: { events: number; templates: number }) => {
    setStats(prev => ({
      ...prev,
      logs: prev.logs + ingestStats.events,
      templates: prev.templates + ingestStats.templates,
    }));
    // Refresh services
    getServices().then(data => {
      setServices(data.services);
      setStats(prev => ({ ...prev, services: data.services.length }));
    }).catch(console.error);
  };

  const tabs = [
    { id: 'explorer' as Tab, label: 'Log Explorer', icon: Search, description: 'Browse and search logs' },
    { id: 'templates' as Tab, label: 'Templates', icon: FileText, description: 'View log patterns' },
    { id: 'stats' as Tab, label: 'Statistics', icon: BarChart2, description: 'Analytics & insights' },
    { id: 'reports' as Tab, label: 'Reports', icon: TrendingUp, description: 'Full analysis reports' },
    { id: 'semantic' as Tab, label: 'AI Search', icon: Sparkles, description: 'Natural language search' },
    { id: 'ai-chat' as Tab, label: 'AI Chat', icon: MessageCircle, description: 'Test models & analyze' },
    { id: 'anomaly' as Tab, label: 'Anomalies', icon: Shield, description: 'Detect issues' },
    { id: 'ai-model' as Tab, label: 'AI Dashboard', icon: Cpu, description: 'Model & training' },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 flex">
      {/* Sidebar */}
      <aside className={`bg-gradient-to-b from-slate-900 to-slate-800 text-white transition-all duration-300 flex flex-col shadow-xl ${
        sidebarCollapsed ? 'w-16' : 'w-64'
      }`}>
        <div className="p-4 border-b border-slate-700/50 flex items-center gap-3">
          <div className="w-10 h-10 bg-gradient-to-br from-cyan-400 via-blue-500 to-purple-600 rounded-xl flex items-center justify-center text-xl shadow-lg">
            🧠
          </div>
          {!sidebarCollapsed && (
            <div>
              <h1 className="font-bold text-lg">LogMind AI</h1>
              <span className={`text-xs ${
                apiStatus === 'online' ? 'text-green-400' :
                apiStatus === 'offline' ? 'text-red-400' : 'text-gray-400'
              }`}>
                {apiStatus === 'online' ? '● Connected' : 
                 apiStatus === 'offline' ? '● Disconnected' : '● Checking...'}
              </span>
            </div>
          )}
        </div>

        <nav className="flex-1 p-2">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`w-full flex items-center gap-3 px-3 py-3 rounded-xl mb-1 transition-all duration-200 ${
                  activeTab === tab.id 
                    ? 'bg-gradient-to-r from-cyan-500 to-blue-600 text-white shadow-lg shadow-blue-500/25' 
                    : 'text-slate-400 hover:bg-slate-700/50 hover:text-white'
                }`}
              >
                <Icon size={20} />
                {!sidebarCollapsed && (
                  <div className="text-left">
                    <div className="text-sm font-medium">{tab.label}</div>
                    <div className="text-xs opacity-70">{tab.description}</div>
                  </div>
                )}
              </button>
            );
          })}
        </nav>

        {!sidebarCollapsed && (
          <div className="p-4 border-t border-slate-700/50">
            <h3 className="text-xs uppercase text-slate-400 mb-3">Statistics</h3>
            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span className="text-slate-400 flex items-center gap-2">
                  <Server size={14} /> Services
                </span>
                <span className="font-medium">{stats.services}</span>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span className="text-slate-400 flex items-center gap-2">
                  <Activity size={14} /> Log Events
                </span>
                <span className="font-medium">{stats.logs.toLocaleString()}</span>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span className="text-slate-400 flex items-center gap-2">
                  <BarChart2 size={14} /> Templates
                </span>
                <span className="font-medium">{stats.templates.toLocaleString()}</span>
              </div>
            </div>
            
            {/* Embedding Progress */}
            {embeddingStats && (
              <div className="mt-4 pt-4 border-t border-slate-600">
                <h3 className="text-xs uppercase text-slate-400 mb-2">AI Training</h3>
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-slate-400 flex items-center gap-2">
                      <Brain size={14} /> Trained
                    </span>
                    <span className="font-medium text-green-400">
                      {embeddingStats.percentage.toFixed(1)}%
                    </span>
                  </div>
                  <div className="w-full bg-slate-600 rounded-full h-2">
                    <div 
                      className="bg-gradient-to-r from-green-500 to-emerald-400 h-2 rounded-full transition-all duration-500"
                      style={{ width: `${embeddingStats.percentage}%` }}
                    />
                  </div>
                  <div className="text-xs text-slate-500">
                    {embeddingStats.embedded_count.toLocaleString()} / {embeddingStats.total_templates.toLocaleString()} templates
                  </div>
                  {embeddingStats.pending_count > 0 && (
                    <div className="text-xs text-yellow-400">
                      {embeddingStats.pending_count.toLocaleString()} pending
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}

        <button
          onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
          className="p-4 border-t border-slate-700/50 text-slate-400 hover:text-white transition-colors"
        >
          <ChevronRight className={`transform transition-transform ${sidebarCollapsed ? '' : 'rotate-180'}`} size={20} />
        </button>
      </aside>

      {/* Main content */}
      <div className="flex-1 flex flex-col">
        <header className="bg-white border-b border-slate-200 px-6 py-4 flex items-center justify-between shadow-sm">
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-2">
              <ServiceSelector
                services={services}
                selectedService={selectedService}
                onServiceChange={setSelectedService}
              />
            </div>

            <div className="flex items-center gap-2">
              <div className="flex items-center gap-3 px-4 py-2.5 bg-white border-2 border-slate-200 rounded-xl shadow-sm hover:border-slate-300 transition-all">
                <div className="p-1.5 bg-emerald-100 rounded-lg">
                  <Clock size={16} className="text-emerald-600" />
                </div>
                <select
                  value={timeRange}
                  onChange={(e) => setTimeRange(e.target.value)}
                  className="bg-transparent text-sm font-medium text-slate-800 focus:outline-none cursor-pointer pr-2"
                >
                  <option value="1h">Last 1 hour</option>
                  <option value="6h">Last 6 hours</option>
                  <option value="24h">Last 24 hours</option>
                  <option value="7d">Last 7 days</option>
                  <option value="30d">Last 30 days</option>
                  <option value="all">All time</option>
                </select>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <input
              ref={fileInputRef}
              type="file"
              accept=".log,.txt,.jsonl,.csv"
              multiple
              onChange={handleFileUpload}
              className="hidden"
              id="log-upload"
            />
            <label
              htmlFor="log-upload"
              className={`flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-emerald-500 to-teal-500 text-white rounded-xl hover:from-emerald-600 hover:to-teal-600 text-sm font-medium transition-all shadow-md shadow-emerald-500/20 cursor-pointer ${
                uploadProgress.length > 0 || apiStatus !== 'online' ? 'opacity-50 cursor-not-allowed' : ''
              }`}
            >
              <Upload size={16} />
              Upload
            </label>

            <button
              onClick={() => setFileBrowserOpen(true)}
              disabled={loading || apiStatus !== 'online'}
              className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-amber-500 to-orange-500 text-white rounded-xl hover:from-amber-600 hover:to-orange-600 disabled:opacity-50 disabled:cursor-not-allowed text-sm font-medium transition-all shadow-md shadow-amber-500/20"
            >
              <FolderOpen size={16} />
              Browse Logs
            </button>

            <button
              onClick={handleEmbed}
              disabled={isTraining || apiStatus !== 'online' || (embeddingStats?.pending_count === 0)}
              className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-violet-500 via-purple-500 to-fuchsia-500 text-white rounded-xl hover:from-violet-600 hover:via-purple-600 hover:to-fuchsia-600 disabled:opacity-50 disabled:cursor-not-allowed text-sm font-medium transition-all shadow-md shadow-purple-500/20"
            >
              <Brain size={16} className={isTraining ? 'animate-pulse' : ''} />
              {isTraining ? (
                <>Training... {embeddingStats?.percentage.toFixed(0)}%</>
              ) : embeddingStats?.pending_count === 0 ? (
                <>Trained ✓</>
              ) : (
                <>Train AI ({embeddingStats?.pending_count.toLocaleString()} pending)</>
              )}
            </button>
          </div>
        </header>

        {/* Upload Progress */}
        {uploadProgress.length > 0 && (
          <div className="bg-blue-50 border-b border-blue-200 px-6 py-3">
            <div className="text-sm font-medium text-blue-800 mb-2">Uploading Files</div>
            <div className="space-y-2">
              {uploadProgress.map((up, idx) => (
                <div key={idx} className="flex items-center gap-3">
                  <span className="text-sm text-blue-700 w-48 truncate">{up.fileName}</span>
                  <div className="flex-1 h-2 bg-blue-200 rounded-full overflow-hidden">
                    <div 
                      className={`h-full transition-all duration-300 ${
                        up.status === 'error' ? 'bg-red-500' :
                        up.status === 'done' ? 'bg-green-500' : 'bg-blue-600'
                      }`}
                      style={{ width: `${up.progress}%` }}
                    />
                  </div>
                  <span className="text-xs text-blue-600 w-20">
                    {up.status === 'uploading' && `${up.progress}%`}
                    {up.status === 'processing' && 'Processing...'}
                    {up.status === 'done' && '✓ Done'}
                    {up.status === 'error' && '✗ Error'}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {error && (
          <div className="bg-red-50 border-b border-red-200 px-6 py-3 flex items-center justify-between">
            <div className="flex items-center gap-2 text-red-700">
              <AlertCircle size={16} />
              <span className="text-sm">{error}</span>
            </div>
            <button 
              onClick={() => setError(null)} 
              className="text-red-600 hover:text-red-800 text-sm underline"
            >
              Dismiss
            </button>
          </div>
        )}

        <main className="flex-1 overflow-hidden relative">
          {/* Keep all tabs mounted but only show the active one */}
          <div className={`absolute inset-0 ${activeTab === 'explorer' ? '' : 'hidden'}`}>
            <LogExplorer serviceName={selectedService} timeRange={timeRange} />
          </div>
          <div className={`absolute inset-0 ${activeTab === 'templates' ? '' : 'hidden'}`}>
            <TemplatesView serviceName={selectedService} timeRange={timeRange} />
          </div>
          <div className={`absolute inset-0 ${activeTab === 'stats' ? '' : 'hidden'}`}>
            <StatsView serviceName={selectedService} timeRange={timeRange} />
          </div>
          <div className={`absolute inset-0 ${activeTab === 'semantic' ? '' : 'hidden'}`}>
            <SemanticSearchView serviceName={selectedService} timeRange={timeRange} />
          </div>
          <div className={`absolute inset-0 ${activeTab === 'anomaly' ? '' : 'hidden'}`}>
            <AnomalyView serviceName={selectedService} timeRange={timeRange} />
          </div>
          <div className={`absolute inset-0 ${activeTab === 'ai-model' ? '' : 'hidden'}`}>
            <AIModelView />
          </div>
          <div className={`absolute inset-0 ${activeTab === 'ai-chat' ? '' : 'hidden'}`}>
            <AIChat />
          </div>
          <div className={`absolute inset-0 ${activeTab === 'reports' ? '' : 'hidden'}`}>
            <ReportsView serviceName={selectedService} timeRange={timeRange} />
          </div>
        </main>
      </div>

      {/* File Browser Modal */}
      <FileBrowser
        isOpen={fileBrowserOpen}
        onClose={() => setFileBrowserOpen(false)}
        onIngestComplete={handleFileBrowserIngest}
      />
    </div>
  );
}
