'use client';

import { useState, useEffect } from 'react';
import { 
  Folder, FileText, ChevronLeft, Check, X, Upload as UploadIcon,
  RefreshCw, FolderOpen
} from 'lucide-react';
import { browseLogFiles, ingestSelectedFiles, FileInfo } from '@/lib/api';

interface FileBrowserProps {
  isOpen: boolean;
  onClose: () => void;
  onIngestComplete: (stats: { events: number; templates: number }) => void;
}

function formatSize(bytes: number): string {
  if (bytes === 0) return '-';
  const units = ['B', 'KB', 'MB', 'GB'];
  let size = bytes;
  let unitIndex = 0;
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex++;
  }
  return `${size.toFixed(1)} ${units[unitIndex]}`;
}

export default function FileBrowser({ isOpen, onClose, onIngestComplete }: FileBrowserProps) {
  const [currentPath, setCurrentPath] = useState('');
  const [files, setFiles] = useState<FileInfo[]>([]);
  const [selectedFiles, setSelectedFiles] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(false);
  const [ingesting, setIngesting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [ingestProgress, setIngestProgress] = useState<string>('');

  const loadFiles = async (path: string = '') => {
    setLoading(true);
    setError(null);
    try {
      const data = await browseLogFiles(path || undefined);
      setCurrentPath(data.current_path);
      setFiles(data.files);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load files');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (isOpen) {
      loadFiles();
      setSelectedFiles(new Set());
    }
  }, [isOpen]);

  const handleNavigate = (file: FileInfo) => {
    if (file.is_dir) {
      loadFiles(file.path);
      setSelectedFiles(new Set());
    }
  };

  const handleGoUp = () => {
    const parts = currentPath.split('/').filter(Boolean);
    parts.pop();
    loadFiles(parts.join('/'));
    setSelectedFiles(new Set());
  };

  const toggleSelect = (file: FileInfo) => {
    if (file.is_dir) return;
    
    const newSelected = new Set(selectedFiles);
    if (newSelected.has(file.path)) {
      newSelected.delete(file.path);
    } else {
      newSelected.add(file.path);
    }
    setSelectedFiles(newSelected);
  };

  const selectAll = () => {
    const allFiles = files.filter(f => !f.is_dir).map(f => f.path);
    setSelectedFiles(new Set(allFiles));
  };

  const selectNone = () => {
    setSelectedFiles(new Set());
  };

  const handleIngest = async () => {
    if (selectedFiles.size === 0) return;
    
    setIngesting(true);
    setIngestProgress(`Ingesting ${selectedFiles.size} files...`);
    setError(null);
    
    try {
      const result = await ingestSelectedFiles(Array.from(selectedFiles));
      setIngestProgress(`Done! ${result.stats.events_inserted} events, ${result.stats.templates_discovered} templates`);
      onIngestComplete({
        events: result.stats.events_inserted,
        templates: result.stats.templates_discovered,
      });
      setTimeout(() => {
        setSelectedFiles(new Set());
        setIngestProgress('');
      }, 2000);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Ingestion failed');
      setIngestProgress('');
    } finally {
      setIngesting(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-white rounded-xl shadow-2xl w-[700px] max-h-[80vh] flex flex-col">
        {/* Header */}
        <div className="px-6 py-4 border-b border-gray-200 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <FolderOpen className="text-blue-600" size={24} />
            <div>
              <h2 className="text-lg font-semibold">Browse Log Files</h2>
              <p className="text-sm text-gray-500">
                {currentPath ? `Logs/${currentPath}` : 'Logs/'}
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <X size={20} />
          </button>
        </div>

        {/* Toolbar */}
        <div className="px-6 py-3 border-b border-gray-100 flex items-center justify-between bg-gray-50">
          <div className="flex items-center gap-2">
            <button
              onClick={handleGoUp}
              disabled={!currentPath || loading}
              className="flex items-center gap-1 px-3 py-1.5 text-sm bg-white border border-gray-200 rounded-lg hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <ChevronLeft size={16} />
              Back
            </button>
            <button
              onClick={() => loadFiles(currentPath)}
              disabled={loading}
              className="p-1.5 text-gray-500 hover:bg-white rounded-lg"
            >
              <RefreshCw size={16} className={loading ? 'animate-spin' : ''} />
            </button>
          </div>
          <div className="flex items-center gap-2 text-sm">
            <button onClick={selectAll} className="text-blue-600 hover:underline">
              Select All
            </button>
            <span className="text-gray-300">|</span>
            <button onClick={selectNone} className="text-blue-600 hover:underline">
              Select None
            </button>
            <span className="text-gray-400 ml-2">
              {selectedFiles.size} selected
            </span>
          </div>
        </div>

        {/* File List */}
        <div className="flex-1 overflow-auto p-4">
          {error && (
            <div className="bg-red-50 text-red-700 px-4 py-3 rounded-lg mb-4">
              {error}
            </div>
          )}
          
          {loading ? (
            <div className="flex items-center justify-center py-12 text-gray-500">
              <RefreshCw className="animate-spin mr-2" size={20} />
              Loading...
            </div>
          ) : files.length === 0 ? (
            <div className="text-center py-12 text-gray-500">
              No log files found in this directory
            </div>
          ) : (
            <div className="space-y-1">
              {files.map((file) => (
                <div
                  key={file.path}
                  onClick={() => file.is_dir ? handleNavigate(file) : toggleSelect(file)}
                  className={`flex items-center gap-3 px-4 py-2.5 rounded-lg cursor-pointer transition-colors ${
                    selectedFiles.has(file.path)
                      ? 'bg-blue-50 border border-blue-200'
                      : 'hover:bg-gray-50 border border-transparent'
                  }`}
                >
                  <div className={`w-5 h-5 flex items-center justify-center ${
                    file.is_dir ? '' : 'border rounded'
                  } ${
                    selectedFiles.has(file.path) 
                      ? 'bg-blue-600 border-blue-600 text-white' 
                      : 'border-gray-300'
                  }`}>
                    {file.is_dir ? (
                      <Folder size={20} className="text-amber-500" />
                    ) : selectedFiles.has(file.path) ? (
                      <Check size={14} />
                    ) : null}
                  </div>
                  
                  <div className="flex-1 flex items-center gap-2">
                    {!file.is_dir && <FileText size={16} className="text-gray-400" />}
                    <span className={file.is_dir ? 'font-medium' : ''}>
                      {file.name}
                    </span>
                  </div>
                  
                  <span className="text-sm text-gray-400">
                    {formatSize(file.size)}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-gray-200 flex items-center justify-between bg-gray-50">
          {ingestProgress ? (
            <span className="text-sm text-green-600 font-medium">{ingestProgress}</span>
          ) : (
            <span className="text-sm text-gray-500">
              Select files to ingest into the database
            </span>
          )}
          <div className="flex items-center gap-3">
            <button
              onClick={onClose}
              className="px-4 py-2 text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={handleIngest}
              disabled={selectedFiles.size === 0 || ingesting}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <UploadIcon size={16} />
              {ingesting ? 'Ingesting...' : `Ingest ${selectedFiles.size} Files`}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
