'use client';

import React, { useEffect, useState } from 'react';
import { useConnection, useConnectionActions } from '@/lib/store';

interface ConnectionStatusProps {
  className?: string;
  compact?: boolean;
}

/**
 * Displays real-time connection status for API and Ollama.
 * Automatically checks connections on mount and periodically.
 */
export function ConnectionStatus({ className = '', compact = false }: ConnectionStatusProps) {
  const connection = useConnection();
  const { checkConnections } = useConnectionActions();
  const [isInitialized, setIsInitialized] = useState(false);

  useEffect(() => {
    // Initial check
    checkConnections().then(() => setIsInitialized(true));
    
    // Check every 30 seconds
    const interval = setInterval(() => {
      checkConnections();
    }, 30000);
    
    return () => clearInterval(interval);
  }, [checkConnections]);

  const getStatusColor = (status: 'connected' | 'disconnected' | 'checking') => {
    switch (status) {
      case 'connected':
        return 'bg-green-500';
      case 'disconnected':
        return 'bg-red-500';
      case 'checking':
        return 'bg-yellow-500 animate-pulse';
    }
  };

  const getStatusText = (status: 'connected' | 'disconnected' | 'checking') => {
    switch (status) {
      case 'connected':
        return 'Connected';
      case 'disconnected':
        return 'Disconnected';
      case 'checking':
        return 'Checking...';
    }
  };

  if (!isInitialized && connection.api === 'checking') {
    return (
      <div className={`flex items-center gap-2 ${className}`}>
        <div className="w-2 h-2 rounded-full bg-gray-400 animate-pulse" />
        <span className="text-xs text-gray-500">Connecting...</span>
      </div>
    );
  }

  if (compact) {
    return (
      <div className={`flex items-center gap-1.5 ${className}`}>
        <div className={`w-2 h-2 rounded-full ${getStatusColor(connection.api)}`} />
        {connection.api !== 'connected' && (
          <span className="text-xs text-red-500 font-medium">API Offline</span>
        )}
        {connection.api === 'connected' && connection.ollama !== 'connected' && (
          <>
            <div className={`w-2 h-2 rounded-full ${getStatusColor(connection.ollama)}`} />
            <span className="text-xs text-yellow-600 font-medium">LLM Offline</span>
          </>
        )}
      </div>
    );
  }

  return (
    <div className={`bg-gray-800 rounded-lg p-3 ${className}`}>
      <h4 className="text-xs font-semibold text-gray-400 uppercase mb-2">
        System Status
      </h4>
      
      <div className="space-y-2">
        {/* API Status */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${getStatusColor(connection.api)}`} />
            <span className="text-sm text-gray-300">API Server</span>
          </div>
          <span className={`text-xs font-medium ${
            connection.api === 'connected' ? 'text-green-400' : 
            connection.api === 'disconnected' ? 'text-red-400' : 'text-yellow-400'
          }`}>
            {getStatusText(connection.api)}
          </span>
        </div>
        
        {/* Ollama Status */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${getStatusColor(connection.ollama)}`} />
            <span className="text-sm text-gray-300">Ollama LLM</span>
          </div>
          <span className={`text-xs font-medium ${
            connection.ollama === 'connected' ? 'text-green-400' : 
            connection.ollama === 'disconnected' ? 'text-red-400' : 'text-yellow-400'
          }`}>
            {getStatusText(connection.ollama)}
          </span>
        </div>
      </div>
      
      {/* Warning message if Ollama is down */}
      {connection.ollama === 'disconnected' && connection.api === 'connected' && (
        <div className="mt-3 p-2 bg-yellow-900/30 border border-yellow-700 rounded text-xs text-yellow-300">
          <strong>⚠️ Chat Unavailable:</strong> Ollama is not running. 
          Start it with <code className="bg-yellow-800/50 px-1 rounded">ollama serve</code>
        </div>
      )}
      
      {/* Error message if API is down */}
      {connection.api === 'disconnected' && (
        <div className="mt-3 p-2 bg-red-900/30 border border-red-700 rounded text-xs text-red-300">
          <strong>🔴 API Offline:</strong> Cannot connect to LogMind API. 
          Check if the server is running.
        </div>
      )}
      
      {/* Last checked timestamp */}
      {connection.lastChecked && (
        <div className="mt-2 text-xs text-gray-500 text-right">
          Last checked: {new Date(connection.lastChecked).toLocaleTimeString()}
        </div>
      )}
    </div>
  );
}

/**
 * Minimal status indicator for use in headers/navbars.
 */
export function StatusDot() {
  const connection = useConnection();
  const { checkConnections } = useConnectionActions();

  useEffect(() => {
    checkConnections();
    const interval = setInterval(checkConnections, 30000);
    return () => clearInterval(interval);
  }, [checkConnections]);

  const allConnected = connection.api === 'connected' && connection.ollama === 'connected';
  const anyChecking = connection.api === 'checking' || connection.ollama === 'checking';
  const apiDown = connection.api === 'disconnected';

  return (
    <div className="relative group cursor-pointer">
      <div className={`w-2.5 h-2.5 rounded-full ${
        anyChecking ? 'bg-yellow-500 animate-pulse' :
        allConnected ? 'bg-green-500' :
        apiDown ? 'bg-red-500' : 'bg-yellow-500'
      }`} />
      
      {/* Tooltip */}
      <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 bg-gray-900 text-white text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none">
        {anyChecking ? 'Checking connections...' :
         allConnected ? 'All systems operational' :
         apiDown ? 'API server offline' :
         'Ollama LLM offline'}
      </div>
    </div>
  );
}

export default ConnectionStatus;
