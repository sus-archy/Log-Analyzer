'use client';

import { useState, useEffect } from 'react';
import { 
  Brain, Cpu, Activity, CheckCircle, XCircle, Clock, 
  Loader2, RefreshCw, Zap, Database, TrendingUp,
  BarChart2, AlertTriangle, Server, Sparkles
} from 'lucide-react';
import { getEmbeddingStats, processEmbeddings, healthCheck, getModelTrainingStatus, type EmbeddingStats, type ModelTrainingStatus } from '@/lib/api';

interface ModelInfo {
  name: string;
  type: 'embedding' | 'chat';
  status: 'online' | 'offline' | 'loading';
  provider: string;
  parameters?: string;
  contextWindow?: string;
}

interface TrainingMetrics {
  severityAccuracy: number;
  domainAccuracy: number;
  securityAccuracy: number;
  anomalyThreshold: number;
  coveragePercent: number;
  modelsLoaded: boolean;
  lastTrained?: string;
  samplesTrained?: number;
}

export default function AIModelView() {
  const [embeddingStats, setEmbeddingStats] = useState<EmbeddingStats | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [metrics, setMetrics] = useState<TrainingMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [ollamaStatus, setOllamaStatus] = useState<'online' | 'offline' | 'checking'>('checking');
  const [modelStatus, setModelStatus] = useState<ModelTrainingStatus | null>(null);

  const loadData = async () => {
    setLoading(true);
    try {
      // Check Ollama status
      try {
        const response = await fetch('http://127.0.0.1:11434/api/tags');
        if (response.ok) {
          setOllamaStatus('online');
          const data = await response.json();
          const ollamaModels: ModelInfo[] = data.models?.map((m: any) => ({
            name: m.name,
            type: m.name.includes('embed') ? 'embedding' : 'chat',
            status: 'online' as const,
            provider: 'Ollama',
            parameters: m.details?.parameter_size || 'Unknown',
            contextWindow: m.details?.context_length?.toString() || 'Unknown',
          })) || [];
          setModels(ollamaModels);
        } else {
          setOllamaStatus('offline');
        }
      } catch {
        setOllamaStatus('offline');
      }

      // Get embedding stats
      const stats = await getEmbeddingStats();
      setEmbeddingStats(stats);

      // Get real model training status from API
      try {
        const mlStatus = await getModelTrainingStatus();
        setModelStatus(mlStatus);
        
        // Extract real metrics from trained models
        const severityAcc = mlStatus.models.log_classifier.training_stats?.severity_accuracy || 0;
        const domainAcc = mlStatus.models.log_classifier.training_stats?.domain_accuracy || 0;
        const securityAcc = mlStatus.models.security_detector.training_stats?.accuracy || 0;
        const threshold = mlStatus.models.anomaly_detector.threshold || 0.5;
        
        // Get last training info
        const history = mlStatus.models.anomaly_detector.training_history;
        const lastTraining = history && history.length > 0 ? history[history.length - 1] : null;
        
        setMetrics({
          severityAccuracy: severityAcc * 100,
          domainAccuracy: domainAcc * 100,
          securityAccuracy: securityAcc * 100,
          anomalyThreshold: threshold * 100,
          coveragePercent: stats.percentage,
          modelsLoaded: Object.values(mlStatus.models_exist).every(v => v),
          lastTrained: lastTraining?.timestamp,
          samplesTrained: lastTraining?.samples,
        });
      } catch (mlError) {
        console.warn('Could not fetch model training status:', mlError);
        // Fallback to default metrics
        setMetrics({
          severityAccuracy: 0,
          domainAccuracy: 0,
          securityAccuracy: 0,
          anomalyThreshold: 50,
          coveragePercent: stats.percentage,
          modelsLoaded: false,
        });
      }

    } catch (e) {
      console.error('Failed to load AI data:', e);
    } finally {
      setLoading(false);
    }
  };

  const handleTrain = async () => {
    if (isTraining) return;
    setIsTraining(true);
    setTrainingProgress(0);

    try {
      let totalProcessed = 0;
      let consecutiveZeros = 0;
      let retryCount = 0;
      const maxRetries = 3;

      // Keep training until we get 3 consecutive zero responses or hit retry limit
      while (consecutiveZeros < 3 && retryCount < maxRetries) {
        try {
          const result = await processEmbeddings(100); // Process 100 at a time
          const processed = result.processed;
          totalProcessed += processed;
          
          const stats = await getEmbeddingStats();
          setEmbeddingStats(stats);
          setTrainingProgress(stats.percentage);

          if (processed > 0) {
            consecutiveZeros = 0;
            retryCount = 0;
            // Small delay to not overwhelm the server
            await new Promise(resolve => setTimeout(resolve, 300));
          } else {
            consecutiveZeros++;
            // Check if there are still pending templates
            if (stats.pending_count > 0) {
              // There are pending templates but none processed - might be rate limited or error
              await new Promise(resolve => setTimeout(resolve, 2000)); // Wait longer
              retryCount++;
            } else {
              // No more pending templates - we're done
              break;
            }
          }
        } catch (e) {
          console.error('Training batch error:', e);
          retryCount++;
          await new Promise(resolve => setTimeout(resolve, 3000)); // Wait before retry
        }
      }

      console.log(`Training complete: ${totalProcessed} templates processed`);

    } catch (e) {
      console.error('Training error:', e);
    } finally {
      setIsTraining(false);
      loadData();
    }
  };

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 10000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="h-full flex items-center justify-center bg-gradient-to-br from-slate-900 to-slate-800">
        <div className="text-center">
          <Brain size={64} className="animate-pulse mx-auto text-purple-400 mb-4" />
          <p className="text-slate-300 font-medium">Loading AI Systems...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full overflow-auto bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="w-14 h-14 bg-gradient-to-br from-purple-500 via-fuchsia-500 to-pink-500 rounded-2xl flex items-center justify-center shadow-lg shadow-purple-500/30">
              <Brain size={28} className="text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-white">AI Model Dashboard</h1>
              <p className="text-slate-400 mt-0.5">Neural network status & training metrics</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <div className={`flex items-center gap-2 px-4 py-2 rounded-xl ${
              ollamaStatus === 'online' 
                ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' 
                : 'bg-red-500/20 text-red-400 border border-red-500/30'
            }`}>
              <Server size={16} />
              <span className="text-sm font-medium">
                Ollama: {ollamaStatus === 'online' ? 'Connected' : 'Disconnected'}
              </span>
            </div>
            <button
              onClick={loadData}
              className="p-3 bg-slate-700/50 hover:bg-slate-700 rounded-xl text-slate-300 transition-all border border-slate-600"
            >
              <RefreshCw size={18} />
            </button>
          </div>
        </div>

        {/* Model Cards */}
        <div className="grid grid-cols-3 gap-4">
          {models.length > 0 ? models.slice(0, 3).map((model, idx) => (
            <div key={idx} className="bg-slate-800/50 rounded-2xl p-5 border border-slate-700/50 backdrop-blur">
              <div className="flex items-start justify-between mb-4">
                <div className={`p-3 rounded-xl ${
                  model.type === 'embedding' 
                    ? 'bg-cyan-500/20 text-cyan-400' 
                    : 'bg-purple-500/20 text-purple-400'
                }`}>
                  {model.type === 'embedding' ? <Database size={24} /> : <Sparkles size={24} />}
                </div>
                <span className={`px-2 py-1 rounded-lg text-xs font-medium ${
                  model.status === 'online' 
                    ? 'bg-emerald-500/20 text-emerald-400' 
                    : 'bg-red-500/20 text-red-400'
                }`}>
                  {model.status}
                </span>
              </div>
              <h3 className="font-bold text-white text-lg mb-1">{model.name}</h3>
              <p className="text-slate-400 text-sm mb-3">{model.provider}</p>
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div className="bg-slate-900/50 rounded-lg p-2">
                  <span className="text-slate-500">Params</span>
                  <p className="text-slate-200 font-medium">{model.parameters}</p>
                </div>
                <div className="bg-slate-900/50 rounded-lg p-2">
                  <span className="text-slate-500">Context</span>
                  <p className="text-slate-200 font-medium">{model.contextWindow}</p>
                </div>
              </div>
            </div>
          )) : (
            <div className="col-span-3 bg-slate-800/30 rounded-2xl p-8 text-center border border-slate-700/50">
              <AlertTriangle size={48} className="mx-auto text-amber-400 mb-4" />
              <p className="text-slate-300">No models detected. Make sure Ollama is running.</p>
            </div>
          )}
        </div>

        {/* Training Status */}
        <div className="grid grid-cols-2 gap-6">
          {/* Embedding Progress */}
          <div className="bg-slate-800/50 rounded-2xl p-6 border border-slate-700/50">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-lg font-bold text-white flex items-center gap-2">
                <Zap className="text-amber-400" size={20} />
                Embedding Training
              </h2>
              <button
                onClick={handleTrain}
                disabled={isTraining || !embeddingStats || embeddingStats.pending_count === 0}
                className="px-4 py-2 bg-gradient-to-r from-purple-500 to-fuchsia-500 text-white rounded-xl hover:from-purple-600 hover:to-fuchsia-600 disabled:opacity-50 disabled:cursor-not-allowed text-sm font-medium transition-all flex items-center gap-2"
              >
                {isTraining ? (
                  <>
                    <Loader2 size={16} className="animate-spin" />
                    Training...
                  </>
                ) : (
                  <>
                    <Brain size={16} />
                    {embeddingStats?.pending_count === 0 ? 'Fully Trained' : 'Start Training'}
                  </>
                )}
              </button>
            </div>

            {embeddingStats && (
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between text-sm mb-2">
                    <span className="text-slate-400">Progress</span>
                    <span className="text-white font-bold">{embeddingStats.percentage.toFixed(1)}%</span>
                  </div>
                  <div className="h-4 bg-slate-700 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-gradient-to-r from-purple-500 via-fuchsia-500 to-pink-500 transition-all duration-500 relative"
                      style={{ width: `${embeddingStats.percentage}%` }}
                    >
                      <div className="absolute inset-0 bg-white/20 animate-pulse" />
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-4 gap-3">
                  <div className="bg-slate-900/50 rounded-xl p-3 text-center">
                    <p className="text-2xl font-bold text-white">{embeddingStats.total_templates.toLocaleString()}</p>
                    <p className="text-xs text-slate-400">Total</p>
                  </div>
                  <div className="bg-slate-900/50 rounded-xl p-3 text-center">
                    <p className="text-2xl font-bold text-emerald-400">{embeddingStats.embedded_count.toLocaleString()}</p>
                    <p className="text-xs text-slate-400">Trained</p>
                  </div>
                  <div className="bg-slate-900/50 rounded-xl p-3 text-center">
                    <p className="text-2xl font-bold text-amber-400">{embeddingStats.pending_count.toLocaleString()}</p>
                    <p className="text-xs text-slate-400">Pending</p>
                  </div>
                  <div className="bg-slate-900/50 rounded-xl p-3 text-center">
                    <p className="text-2xl font-bold text-red-400">{embeddingStats.failed_count.toLocaleString()}</p>
                    <p className="text-xs text-slate-400">Failed</p>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Model Metrics */}
          <div className="bg-slate-800/50 rounded-2xl p-6 border border-slate-700/50">
            <h2 className="text-lg font-bold text-white flex items-center gap-2 mb-6">
              <TrendingUp className="text-cyan-400" size={20} />
              Model Performance
              {metrics?.modelsLoaded && (
                <span className="ml-auto flex items-center gap-1 text-xs bg-emerald-500/20 text-emerald-400 px-2 py-1 rounded-full">
                  <CheckCircle size={12} />
                  Models Loaded
                </span>
              )}
            </h2>

            {metrics && (
              <div className="space-y-4">
                {/* Classifier Metrics */}
                <div className="space-y-3">
                  <div className="text-xs text-slate-500 uppercase tracking-wider font-medium">Log Classifier</div>
                  {[
                    { label: 'Severity Classification', value: metrics.severityAccuracy, color: 'from-emerald-500 to-teal-500', icon: '📊' },
                    { label: 'Domain Classification', value: metrics.domainAccuracy, color: 'from-blue-500 to-cyan-500', icon: '🏷️' },
                  ].map((metric) => (
                    <div key={metric.label}>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="text-slate-400 flex items-center gap-1.5">
                          <span>{metric.icon}</span>
                          {metric.label}
                        </span>
                        <span className={`font-bold ${metric.value >= 90 ? 'text-emerald-400' : metric.value >= 70 ? 'text-amber-400' : 'text-red-400'}`}>
                          {metric.value.toFixed(2)}%
                        </span>
                      </div>
                      <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
                        <div 
                          className={`h-full bg-gradient-to-r ${metric.color} transition-all duration-1000`}
                          style={{ width: `${Math.min(100, metric.value)}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>

                {/* Security Detector */}
                <div className="space-y-3 pt-2">
                  <div className="text-xs text-slate-500 uppercase tracking-wider font-medium">Security Detector</div>
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-slate-400 flex items-center gap-1.5">
                        <span>🔒</span>
                        Threat Detection
                      </span>
                      <span className={`font-bold ${metrics.securityAccuracy >= 90 ? 'text-emerald-400' : metrics.securityAccuracy >= 70 ? 'text-amber-400' : 'text-red-400'}`}>
                        {metrics.securityAccuracy.toFixed(2)}%
                      </span>
                    </div>
                    <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-gradient-to-r from-purple-500 to-fuchsia-500 transition-all duration-1000"
                        style={{ width: `${Math.min(100, metrics.securityAccuracy)}%` }}
                      />
                    </div>
                  </div>
                </div>

                {/* Anomaly Detector */}
                <div className="space-y-3 pt-2">
                  <div className="text-xs text-slate-500 uppercase tracking-wider font-medium">Anomaly Detector</div>
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-slate-400 flex items-center gap-1.5">
                        <span>⚠️</span>
                        Detection Threshold
                      </span>
                      <span className="text-amber-400 font-bold">
                        {metrics.anomalyThreshold.toFixed(1)}%
                      </span>
                    </div>
                    <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-gradient-to-r from-amber-500 to-orange-500 transition-all duration-1000"
                        style={{ width: `${Math.min(100, metrics.anomalyThreshold)}%` }}
                      />
                    </div>
                  </div>
                </div>

                {/* Additional Info */}
                <div className="mt-6 pt-4 border-t border-slate-700 space-y-2">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-slate-400">Embedding Coverage</span>
                    <span className="text-emerald-400 font-medium">{metrics.coveragePercent.toFixed(1)}%</span>
                  </div>
                  {metrics.samplesTrained && (
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-slate-400">Training Samples</span>
                      <span className="text-white font-medium">{metrics.samplesTrained.toLocaleString()}</span>
                    </div>
                  )}
                  {metrics.lastTrained && (
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-slate-400">Last Trained</span>
                      <span className="text-slate-300 font-medium">
                        {new Date(metrics.lastTrained).toLocaleDateString()}
                      </span>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* System Info */}
        <div className="bg-slate-800/30 rounded-2xl p-6 border border-slate-700/50">
          <h2 className="text-lg font-bold text-white flex items-center gap-2 mb-4">
            <Cpu className="text-blue-400" size={20} />
            System Configuration
          </h2>
          <div className="grid grid-cols-4 gap-4">
            <div className="bg-slate-900/50 rounded-xl p-4">
              <p className="text-slate-400 text-sm">Vector Database</p>
              <p className="text-white font-bold text-lg mt-1">FAISS</p>
              <p className="text-slate-500 text-xs">In-Memory Index</p>
            </div>
            <div className="bg-slate-900/50 rounded-xl p-4">
              <p className="text-slate-400 text-sm">Embedding Model</p>
              <p className="text-white font-bold text-lg mt-1">nomic-embed-text</p>
              <p className="text-slate-500 text-xs">768 dimensions</p>
            </div>
            <div className="bg-slate-900/50 rounded-xl p-4">
              <p className="text-slate-400 text-sm">LLM Provider</p>
              <p className="text-white font-bold text-lg mt-1">Ollama</p>
              <p className="text-slate-500 text-xs">Local inference</p>
            </div>
            <div className="bg-slate-900/50 rounded-xl p-4">
              <p className="text-slate-400 text-sm">Backend</p>
              <p className="text-white font-bold text-lg mt-1">FastAPI</p>
              <p className="text-slate-500 text-xs">Async Python</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
