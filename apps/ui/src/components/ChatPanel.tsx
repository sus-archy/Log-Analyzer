'use client';

import { useState } from 'react';
import { Send, Loader2, FileText, Hash, AlertCircle } from 'lucide-react';
import { chat, getTimeRange } from '@/lib/api';
import type { ChatResponse, Citation } from '@/types';

interface Props {
  serviceName: string;
  timeRange: string;
}

interface Message {
  role: 'user' | 'assistant';
  content: string;
  citations?: Citation[];
  confidence?: string;
  nextSteps?: string[];
}

export default function ChatPanel({ serviceName, timeRange }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const suggestedPrompts = [
    "What are the most common errors?",
    "Why are there connection failures?",
    "Summarize the issues in this time window",
    "What patterns indicate performance problems?",
  ];

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const question = input.trim();
    setInput('');
    setError(null);

    // Add user message
    setMessages((prev) => [...prev, { role: 'user', content: question }]);

    setLoading(true);

    try {
      const { from, to } = getTimeRange(timeRange);
      const response = await chat({
        service_name: serviceName || '',
        from,
        to,
        question,
      });

      // Add assistant message
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: response.answer,
          citations: response.citations,
          confidence: response.confidence,
          nextSteps: response.next_steps,
        },
      ]);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Chat failed');
    } finally {
      setLoading(false);
    }
  };

  const handleSuggestedPrompt = (prompt: string) => {
    setInput(prompt);
  };

  return (
    <div className="h-full flex flex-col bg-gradient-to-b from-slate-50 to-white">
      {/* Chat header */}
      <div className="p-4 border-b border-slate-200 bg-white/80 backdrop-blur-sm">
        <h2 className="font-semibold text-slate-900">Chat with Logs</h2>
        <p className="text-sm text-slate-500">
          Ask questions about {serviceName ? <><span className="font-medium text-slate-700">{serviceName}</span> logs</> : 'all logs'} in the selected time range
        </p>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-auto p-4 space-y-4">
        {messages.length === 0 ? (
          <div className="text-center py-8">
            <div className="w-16 h-16 bg-gradient-to-br from-violet-500 to-purple-600 rounded-2xl flex items-center justify-center mx-auto mb-4 shadow-lg shadow-purple-500/20">
              <span className="text-2xl">🤖</span>
            </div>
            <p className="text-slate-600 mb-4 font-medium">Ask a question about your logs</p>
            <div className="flex flex-wrap gap-2 justify-center max-w-lg mx-auto">
              {suggestedPrompts.map((prompt) => (
                <button
                  key={prompt}
                  onClick={() => handleSuggestedPrompt(prompt)}
                  className="px-4 py-2 bg-gradient-to-r from-violet-50 to-purple-50 text-purple-700 rounded-xl text-sm hover:from-violet-100 hover:to-purple-100 transition-all border border-purple-200/50 shadow-sm"
                >
                  {prompt}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((msg, i) => (
            <div
              key={i}
              className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-[80%] rounded-2xl p-4 shadow-sm ${
                  msg.role === 'user'
                    ? 'bg-gradient-to-r from-violet-600 to-purple-600 text-white'
                    : 'bg-white text-slate-900 border border-slate-200'
                }`}
              >
                <div className="whitespace-pre-wrap text-sm leading-relaxed">{msg.content}</div>

                {msg.role === 'assistant' && msg.citations && msg.citations.length > 0 && (
                  <div className="mt-4 pt-4 border-t border-gray-200">
                    <p className="text-xs font-medium text-gray-500 mb-2">Citations:</p>
                    <div className="space-y-1">
                      {msg.citations.slice(0, 5).map((citation, j) => (
                        <div
                          key={j}
                          className="flex items-center gap-2 text-xs text-gray-600"
                        >
                          {citation.type === 'template' ? (
                            <>
                              <FileText size={12} />
                              <span className="truncate">
                                Template: {citation.template_text?.substring(0, 50)}...
                              </span>
                              <span className="text-gray-400">
                                <Hash size={10} className="inline" />
                                {citation.template_hash}
                              </span>
                            </>
                          ) : (
                            <>
                              <Hash size={12} />
                              <span>Log ID: {citation.log_id}</span>
                            </>
                          )}
                        </div>
                      ))}
                      {msg.citations.length > 5 && (
                        <p className="text-xs text-gray-400">
                          +{msg.citations.length - 5} more citations
                        </p>
                      )}
                    </div>
                  </div>
                )}

                {msg.role === 'assistant' && msg.confidence && (
                  <div className="mt-2 flex items-center gap-2">
                    <span
                      className={`px-2 py-0.5 rounded text-xs font-medium ${
                        msg.confidence === 'high'
                          ? 'bg-green-100 text-green-700'
                          : msg.confidence === 'medium'
                          ? 'bg-yellow-100 text-yellow-700'
                          : 'bg-red-100 text-red-700'
                      }`}
                    >
                      {msg.confidence} confidence
                    </span>
                  </div>
                )}

                {msg.role === 'assistant' && msg.nextSteps && msg.nextSteps.length > 0 && (
                  <div className="mt-4 pt-4 border-t border-gray-200">
                    <p className="text-xs font-medium text-gray-500 mb-2">Suggested next steps:</p>
                    <ul className="space-y-1">
                      {msg.nextSteps.map((step, j) => (
                        <li
                          key={j}
                          className="text-xs text-gray-600 cursor-pointer hover:text-blue-600"
                          onClick={() => handleSuggestedPrompt(step)}
                        >
                          → {step}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>
          ))
        )}

        {loading && (
          <div className="flex justify-start">
            <div className="bg-white border border-slate-200 rounded-2xl p-4 flex items-center gap-3 text-slate-600 shadow-sm">
              <div className="w-8 h-8 bg-gradient-to-br from-violet-500 to-purple-600 rounded-lg flex items-center justify-center">
                <Loader2 size={16} className="animate-spin text-white" />
              </div>
              <div>
                <p className="font-medium text-sm">Analyzing your logs...</p>
                <p className="text-xs text-slate-400">This may take a moment</p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Error */}
      {error && (
        <div className="px-4 py-3 bg-rose-50 text-rose-700 text-sm flex items-center gap-2 border-t border-rose-200">
          <AlertCircle size={16} />
          <span className="flex-1">{error}</span>
          <button onClick={() => setError(null)} className="text-rose-600 hover:text-rose-800 font-medium">
            Dismiss
          </button>
        </div>
      )}

      {/* Input */}
      <form onSubmit={handleSubmit} className="p-4 border-t border-slate-200 bg-white">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about your logs..."
            disabled={loading}
            className="flex-1 px-4 py-2.5 border border-slate-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-purple-500/50 focus:border-purple-400 disabled:opacity-50 transition-all"
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="px-5 py-2.5 bg-gradient-to-r from-violet-600 to-purple-600 text-white rounded-xl hover:from-violet-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 shadow-md shadow-purple-500/20 transition-all font-medium"
          >
            <Send size={16} />
            Send
          </button>
        </div>
      </form>
    </div>
  );
}
