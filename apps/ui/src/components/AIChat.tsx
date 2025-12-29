'use client';

import { useState, useRef, useEffect } from 'react';
import { 
  Send, Loader2, Bot, User, Sparkles, 
  AlertTriangle, CheckCircle, XCircle, 
  FlaskConical, MessageSquare, Search,
  RefreshCw, Zap, Shield, Activity, Square
} from 'lucide-react';
import { mlChat, getModelTrainingStatus, type MLChatResponse, type ModelTrainingStatus } from '@/lib/api';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  analysis?: any;
  modelStatus?: Record<string, boolean>;
  suggestions?: string[];
  timestamp: Date;
}

export default function AIChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [modelStatus, setModelStatus] = useState<ModelTrainingStatus | null>(null);
  const [fastMode, setFastMode] = useState(true); // Default to fast mode
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  // Load model status on mount
  useEffect(() => {
    loadModelStatus();
  }, []);

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const loadModelStatus = async () => {
    try {
      const status = await getModelTrainingStatus();
      setModelStatus(status);
    } catch (e) {
      console.error('Failed to load model status:', e);
    }
  };

  const quickActions = [
    { label: 'Test Models', icon: FlaskConical, message: 'Test if the models are working' },
    { label: 'Check Status', icon: Activity, message: 'What is the model status?' },
    { label: 'Help', icon: MessageSquare, message: 'What can you do?' },
  ];

  const sampleLogs = [
    { label: 'SSH Attack', log: 'Failed password for invalid user admin from 192.168.1.100 port 22 ssh2' },
    { label: 'SQL Injection', log: "SELECT * FROM users WHERE id='1' OR '1'='1'--" },
    { label: 'Connection Error', log: 'Connection timeout after 30000ms to database server db-master-01' },
    { label: 'Auth Success', log: 'User authentication successful for user john.doe@example.com from IP 10.0.0.50' },
  ];

  const handleSubmit = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!input.trim() || loading) return;
    await sendMessage(input.trim());
  };

  const sendMessage = async (message: string) => {
    setInput('');
    setError(null);

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: message,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMessage]);

    setLoading(true);
    
    // Create abort controller for cancellation
    abortControllerRef.current = new AbortController();

    try {
      const response = await mlChat({ 
        message, 
        mode: 'auto',
        fast_mode: fastMode,
        use_llm: !fastMode,
      });
      
      // Check if request was cancelled
      if (abortControllerRef.current?.signal.aborted) {
        return;
      }

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response.response,
        analysis: response.analysis,
        modelStatus: response.model_status,
        suggestions: response.suggestions,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (e) {
      // Don't show error if request was cancelled
      if (abortControllerRef.current?.signal.aborted) {
        return;
      }
      setError(e instanceof Error ? e.message : 'Chat failed');
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: '❌ Sorry, I encountered an error. Please make sure the API server is running.',
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
      abortControllerRef.current = null;
      inputRef.current?.focus();
    }
  };

  const handleCancel = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      setLoading(false);
      // Remove the last user message since we're cancelling
      setMessages((prev) => prev.slice(0, -1));
    }
  };

  const handleQuickAction = (message: string) => {
    sendMessage(message);
  };

  const handleSampleLog = (log: string) => {
    sendMessage(`Analyze this log: ${log}`);
  };

  const handleSuggestion = (suggestion: string) => {
    sendMessage(suggestion);
  };

  const clearChat = () => {
    setMessages([]);
    setError(null);
  };

  const getModelStatusIcon = (trained: boolean) => {
    return trained ? (
      <CheckCircle className="w-4 h-4 text-green-500" />
    ) : (
      <XCircle className="w-4 h-4 text-red-500" />
    );
  };

  const trainedCount = modelStatus?.models_exist 
    ? Object.values(modelStatus.models_exist).filter(Boolean).length 
    : 0;

  return (
    <div className="h-full flex flex-col bg-gradient-to-br from-slate-50 via-white to-purple-50/30">
      {/* Header */}
      <div className="flex-none p-4 border-b border-slate-200 bg-white/80 backdrop-blur-sm">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-violet-500 to-purple-600 rounded-xl flex items-center justify-center shadow-lg shadow-purple-500/20">
              <Bot className="w-5 h-5 text-white" />
            </div>
            <div>
              <h2 className="font-semibold text-slate-900">AI Chat</h2>
              <p className="text-xs text-slate-500">
                Test models & analyze logs with ML
              </p>
            </div>
          </div>
          
          {/* Model Status Badge */}
          <div className="flex items-center gap-3">
            {/* Fast Mode Toggle */}
            <button
              onClick={() => setFastMode(!fastMode)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium flex items-center gap-2 transition-all ${
                fastMode 
                  ? 'bg-amber-100 text-amber-700 hover:bg-amber-200' 
                  : 'bg-purple-100 text-purple-700 hover:bg-purple-200'
              }`}
              title={fastMode ? 'Fast Mode: Instant responses using built-in knowledge' : 'LLM Mode: Uses AI for intelligent responses (slower)'}
            >
              {fastMode ? <Zap className="w-3.5 h-3.5" /> : <Bot className="w-3.5 h-3.5" />}
              {fastMode ? 'Fast Mode' : 'LLM Mode'}
            </button>
            
            <div className={`px-3 py-1.5 rounded-lg text-xs font-medium flex items-center gap-2 ${
              trainedCount === 3 
                ? 'bg-green-100 text-green-700' 
                : trainedCount > 0 
                  ? 'bg-yellow-100 text-yellow-700'
                  : 'bg-red-100 text-red-700'
            }`}>
              {trainedCount === 3 ? <CheckCircle className="w-3.5 h-3.5" /> : <AlertTriangle className="w-3.5 h-3.5" />}
              {trainedCount}/3 Models Ready
            </div>
            <button
              onClick={() => { loadModelStatus(); clearChat(); }}
              className="p-2 text-slate-500 hover:text-slate-700 hover:bg-slate-100 rounded-lg transition-colors"
              title="Refresh & Clear"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-auto p-4 space-y-4">
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-center">
            <div className="w-20 h-20 bg-gradient-to-br from-violet-500 to-purple-600 rounded-3xl flex items-center justify-center mb-6 shadow-xl shadow-purple-500/30">
              <Sparkles className="w-10 h-10 text-white" />
            </div>
            <h3 className="text-xl font-semibold text-slate-900 mb-2">
              LogMind AI Assistant
            </h3>
            <p className="text-slate-600 mb-6 max-w-md">
              Test your trained ML models, analyze logs for anomalies and security threats, 
              or ask questions about the system.
            </p>

            {/* Quick Actions */}
            <div className="flex flex-wrap gap-2 justify-center mb-6">
              {quickActions.map((action) => (
                <button
                  key={action.label}
                  onClick={() => handleQuickAction(action.message)}
                  className="flex items-center gap-2 px-4 py-2.5 bg-white border border-slate-200 rounded-xl text-sm font-medium text-slate-700 hover:border-purple-300 hover:bg-purple-50 transition-all shadow-sm"
                >
                  <action.icon className="w-4 h-4 text-purple-600" />
                  {action.label}
                </button>
              ))}
            </div>

            {/* Sample Logs */}
            <div className="w-full max-w-2xl">
              <p className="text-xs font-medium text-slate-500 mb-3">Try analyzing these sample logs:</p>
              <div className="grid grid-cols-2 gap-2">
                {sampleLogs.map((sample) => (
                  <button
                    key={sample.label}
                    onClick={() => handleSampleLog(sample.log)}
                    className="text-left p-3 bg-white/80 border border-slate-200 rounded-xl text-xs hover:border-purple-300 hover:bg-purple-50/50 transition-all group"
                  >
                    <span className="font-medium text-slate-700 group-hover:text-purple-700 flex items-center gap-1.5">
                      <Search className="w-3 h-3" />
                      {sample.label}
                    </span>
                    <span className="text-slate-500 line-clamp-1 mt-1">{sample.log}</span>
                  </button>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <>
            {messages.map((msg) => (
              <div
                key={msg.id}
                className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-[85%] rounded-2xl p-4 shadow-sm ${
                    msg.role === 'user'
                      ? 'bg-gradient-to-r from-violet-600 to-purple-600 text-white'
                      : 'bg-white text-slate-900 border border-slate-200'
                  }`}
                >
                  {/* Role Icon */}
                  <div className="flex items-start gap-3">
                    {msg.role === 'assistant' && (
                      <div className="flex-none w-7 h-7 bg-gradient-to-br from-violet-500 to-purple-600 rounded-lg flex items-center justify-center">
                        <Bot className="w-4 h-4 text-white" />
                      </div>
                    )}
                    <div className="flex-1 min-w-0">
                      {/* Content with Markdown-like rendering */}
                      <div className="prose prose-sm max-w-none">
                        {msg.content.split('\n').map((line, i) => {
                          // Handle headers
                          if (line.startsWith('**') && line.endsWith('**')) {
                            return (
                              <p key={i} className="font-semibold text-slate-900 mt-2 first:mt-0">
                                {line.replace(/\*\*/g, '')}
                              </p>
                            );
                          }
                          // Handle list items
                          if (line.startsWith('- ') || line.startsWith('• ')) {
                            return (
                              <p key={i} className={`${msg.role === 'user' ? 'text-white/90' : 'text-slate-700'} ml-2`}>
                                {line}
                              </p>
                            );
                          }
                          // Handle numbered items
                          if (/^\d+\.\s/.test(line)) {
                            return (
                              <p key={i} className={`${msg.role === 'user' ? 'text-white/90' : 'text-slate-700'} ml-2`}>
                                {line}
                              </p>
                            );
                          }
                          // Handle code blocks
                          if (line.includes('`')) {
                            const parts = line.split(/`([^`]+)`/);
                            return (
                              <p key={i} className={msg.role === 'user' ? 'text-white/90' : 'text-slate-700'}>
                                {parts.map((part, j) => 
                                  j % 2 === 1 ? (
                                    <code key={j} className="px-1.5 py-0.5 bg-slate-100 text-purple-700 rounded text-xs font-mono">
                                      {part}
                                    </code>
                                  ) : part
                                )}
                              </p>
                            );
                          }
                          // Handle tables
                          if (line.includes('|')) {
                            return null; // Skip table rendering for now
                          }
                          // Regular text
                          if (line.trim()) {
                            return (
                              <p key={i} className={msg.role === 'user' ? 'text-white/90' : 'text-slate-700'}>
                                {line}
                              </p>
                            );
                          }
                          return <br key={i} />;
                        })}
                      </div>

                      {/* Model Status */}
                      {msg.role === 'assistant' && msg.modelStatus && (
                        <div className="mt-4 pt-3 border-t border-slate-100">
                          <div className="flex items-center gap-4 text-xs">
                            <span className="text-slate-500 font-medium">Models:</span>
                            <div className="flex items-center gap-1.5">
                              {getModelStatusIcon(msg.modelStatus.anomaly_detector)}
                              <span className="text-slate-600">Anomaly</span>
                            </div>
                            <div className="flex items-center gap-1.5">
                              {getModelStatusIcon(msg.modelStatus.log_classifier)}
                              <span className="text-slate-600">Classifier</span>
                            </div>
                            <div className="flex items-center gap-1.5">
                              {getModelStatusIcon(msg.modelStatus.security_detector)}
                              <span className="text-slate-600">Security</span>
                            </div>
                          </div>
                        </div>
                      )}

                      {/* Suggestions */}
                      {msg.role === 'assistant' && msg.suggestions && msg.suggestions.length > 0 && (
                        <div className="mt-4 pt-3 border-t border-slate-100">
                          <p className="text-xs font-medium text-slate-500 mb-2">Try next:</p>
                          <div className="flex flex-wrap gap-2">
                            {msg.suggestions.map((suggestion, i) => (
                              <button
                                key={i}
                                onClick={() => handleSuggestion(suggestion)}
                                className="px-3 py-1.5 bg-purple-50 text-purple-700 rounded-lg text-xs font-medium hover:bg-purple-100 transition-colors"
                              >
                                {suggestion}
                              </button>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                    {msg.role === 'user' && (
                      <div className="flex-none w-7 h-7 bg-white/20 rounded-lg flex items-center justify-center">
                        <User className="w-4 h-4 text-white" />
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}

            {/* Loading indicator */}
            {loading && (
              <div className="flex justify-start">
                <div className="bg-white border border-slate-200 rounded-2xl p-4 flex items-center gap-3 shadow-sm">
                  <div className="w-7 h-7 bg-gradient-to-br from-violet-500 to-purple-600 rounded-lg flex items-center justify-center">
                    <Bot className="w-4 h-4 text-white animate-pulse" />
                  </div>
                  <div className="flex items-center gap-2 text-slate-600">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span className="text-sm">Analyzing with ML models...</span>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </>
        )}
      </div>

      {/* Error Banner */}
      {error && (
        <div className="flex-none mx-4 mb-2 p-3 bg-red-50 border border-red-200 rounded-xl flex items-center gap-2 text-red-700 text-sm">
          <AlertTriangle className="w-4 h-4" />
          {error}
          <button
            onClick={() => setError(null)}
            className="ml-auto text-red-500 hover:text-red-700"
          >
            ×
          </button>
        </div>
      )}

      {/* Input Area */}
      <div className="flex-none p-4 border-t border-slate-200 bg-white/80 backdrop-blur-sm">
        <form onSubmit={handleSubmit} className="flex gap-3">
          <div className="flex-1 relative">
            <input
              ref={inputRef}
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a question or paste a log to analyze..."
              className="w-full px-4 py-3 pr-12 border-2 border-slate-200 rounded-xl text-sm text-slate-900 placeholder-slate-400 focus:outline-none focus:ring-4 focus:ring-purple-500/20 focus:border-purple-400 bg-white transition-all"
              disabled={loading}
            />
            <div className="absolute right-3 top-1/2 -translate-y-1/2 flex items-center gap-1">
              <kbd className="hidden sm:inline-block px-1.5 py-0.5 bg-slate-100 text-slate-500 text-xs rounded">
                Enter
              </kbd>
            </div>
          </div>
          {loading ? (
            <button
              type="button"
              onClick={handleCancel}
              className="px-5 py-3 bg-gradient-to-r from-red-500 to-rose-500 text-white rounded-xl font-medium flex items-center gap-2 hover:from-red-600 hover:to-rose-600 transition-all shadow-lg shadow-red-500/20"
            >
              <Square className="w-4 h-4" />
              <span className="hidden sm:inline">Cancel</span>
            </button>
          ) : (
            <button
              type="submit"
              disabled={!input.trim()}
              className="px-5 py-3 bg-gradient-to-r from-violet-600 to-purple-600 text-white rounded-xl font-medium flex items-center gap-2 hover:from-violet-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg shadow-purple-500/20"
            >
              <Send className="w-4 h-4" />
              <span className="hidden sm:inline">Send</span>
            </button>
          )}
        </form>
        <p className="text-xs text-slate-400 mt-2 text-center">
          {fastMode ? (
            <span className="flex items-center justify-center gap-1">
              <Zap className="w-3 h-3 text-amber-500" />
              <span><strong>Fast Mode</strong> - Instant responses using built-in knowledge base</span>
            </span>
          ) : (
            <span className="flex items-center justify-center gap-1">
              <Bot className="w-3 h-3 text-purple-500" />
              <span><strong>LLM Mode</strong> - AI-powered responses (may take 15-30 seconds)</span>
            </span>
          )}
        </p>
      </div>
    </div>
  );
}
