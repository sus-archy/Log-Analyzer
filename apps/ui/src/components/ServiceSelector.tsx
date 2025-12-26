'use client';

import { useState, useRef, useEffect } from 'react';
import { Search, X, Server, ChevronDown, Check, FolderOpen } from 'lucide-react';

interface ServiceSelectorProps {
  services: string[];
  selectedService: string;
  onServiceChange: (service: string) => void;
}

// Helper to get a nice display name from service path
function getServiceDisplayName(service: string): string {
  // If it's a path-like name, show just the last part with context
  if (service.includes('/') || service.includes('\\')) {
    const parts = service.split(/[/\\]/);
    const name = parts[parts.length - 1] || parts[parts.length - 2] || service;
    const parent = parts[parts.length - 2];
    return parent ? `${name} (${parent})` : name;
  }
  return service;
}

// Group services by their parent folder
function groupServices(services: string[]): Map<string, string[]> {
  const groups = new Map<string, string[]>();
  
  services.forEach(service => {
    let group = 'Other';
    if (service.includes('/') || service.includes('\\') || service.includes('-')) {
      const parts = service.split(/[/\\-]/);
      if (parts.length > 1) {
        group = parts[0].charAt(0).toUpperCase() + parts[0].slice(1);
      }
    }
    
    if (!groups.has(group)) {
      groups.set(group, []);
    }
    groups.get(group)!.push(service);
  });
  
  return groups;
}

export default function ServiceSelector({ 
  services, 
  selectedService, 
  onServiceChange 
}: ServiceSelectorProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const containerRef = useRef<HTMLDivElement>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);

  // Filter services based on search
  const filteredServices = searchQuery
    ? services.filter(s => 
        s.toLowerCase().includes(searchQuery.toLowerCase()) ||
        getServiceDisplayName(s).toLowerCase().includes(searchQuery.toLowerCase())
      )
    : services;

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Focus search input when opening
  useEffect(() => {
    if (isOpen && searchInputRef.current) {
      searchInputRef.current.focus();
    }
  }, [isOpen]);

  const handleSelect = (service: string) => {
    onServiceChange(service);
    setIsOpen(false);
    setSearchQuery('');
  };

  return (
    <div className="relative" ref={containerRef}>
      {/* Trigger Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`
          flex items-center gap-3 px-4 py-2.5 
          bg-white border-2 rounded-xl 
          text-sm font-medium text-slate-800 
          min-w-[280px] max-w-[320px]
          transition-all duration-200
          hover:border-blue-300 hover:shadow-md
          ${isOpen 
            ? 'border-blue-500 ring-2 ring-blue-500/20 shadow-lg' 
            : 'border-slate-200 shadow-sm'
          }
        `}
      >
        <div className={`p-1.5 rounded-lg ${selectedService ? 'bg-blue-100' : 'bg-slate-100'}`}>
          {selectedService ? (
            <FolderOpen size={16} className="text-blue-600" />
          ) : (
            <Server size={16} className="text-slate-500" />
          )}
        </div>
        <span className="flex-1 text-left truncate">
          {selectedService ? getServiceDisplayName(selectedService) : 'All Services'}
        </span>
        <div className="flex items-center gap-2">
          {selectedService && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                onServiceChange('');
              }}
              className="p-1 hover:bg-slate-100 rounded-full transition-colors"
            >
              <X size={14} className="text-slate-400 hover:text-slate-600" />
            </button>
          )}
          <ChevronDown 
            size={18} 
            className={`text-slate-400 transition-transform duration-200 ${isOpen ? 'rotate-180' : ''}`} 
          />
        </div>
      </button>

      {/* Dropdown */}
      {isOpen && (
        <div className="absolute z-50 top-full left-0 mt-2 w-[360px] bg-white border border-slate-200 rounded-2xl shadow-2xl overflow-hidden animate-in fade-in slide-in-from-top-2 duration-200">
          {/* Search Header */}
          <div className="p-3 border-b border-slate-100 bg-gradient-to-r from-slate-50 to-white">
            <div className="relative">
              <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
              <input
                ref={searchInputRef}
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search services..."
                className="w-full pl-9 pr-4 py-2.5 bg-white border border-slate-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all"
              />
              {searchQuery && (
                <button
                  onClick={() => setSearchQuery('')}
                  className="absolute right-3 top-1/2 -translate-y-1/2 p-1 hover:bg-slate-100 rounded-full"
                >
                  <X size={14} className="text-slate-400" />
                </button>
              )}
            </div>
            <div className="flex items-center justify-between mt-2 px-1">
              <span className="text-xs text-slate-500">
                {filteredServices.length} service{filteredServices.length !== 1 ? 's' : ''} 
                {searchQuery && ` matching "${searchQuery}"`}
              </span>
              {selectedService && (
                <button
                  onClick={() => handleSelect('')}
                  className="text-xs text-blue-600 hover:text-blue-700 font-medium"
                >
                  Clear selection
                </button>
              )}
            </div>
          </div>

          {/* Service List */}
          <div className="max-h-[400px] overflow-y-auto">
            {/* All Services Option */}
            <button
              onClick={() => handleSelect('')}
              className={`
                w-full flex items-center gap-3 px-4 py-3 
                hover:bg-blue-50 transition-colors text-left
                border-b border-slate-100
                ${!selectedService ? 'bg-blue-50' : ''}
              `}
            >
              <div className={`p-2 rounded-lg ${!selectedService ? 'bg-blue-500' : 'bg-slate-100'}`}>
                <Server size={16} className={!selectedService ? 'text-white' : 'text-slate-500'} />
              </div>
              <div className="flex-1">
                <div className={`font-medium ${!selectedService ? 'text-blue-700' : 'text-slate-700'}`}>
                  All Services
                </div>
                <div className="text-xs text-slate-500">
                  View logs from all {services.length} services
                </div>
              </div>
              {!selectedService && (
                <Check size={18} className="text-blue-600" />
              )}
            </button>

            {/* Filtered Services */}
            {filteredServices.length === 0 ? (
              <div className="p-8 text-center">
                <Search size={32} className="mx-auto text-slate-300 mb-2" />
                <p className="text-slate-500 text-sm">No services found</p>
                <p className="text-slate-400 text-xs mt-1">Try a different search term</p>
              </div>
            ) : (
              filteredServices.map((service, index) => {
                const isSelected = service === selectedService;
                const displayName = getServiceDisplayName(service);
                
                return (
                  <button
                    key={service}
                    onClick={() => handleSelect(service)}
                    className={`
                      w-full flex items-center gap-3 px-4 py-2.5
                      hover:bg-blue-50 transition-colors text-left
                      ${index < filteredServices.length - 1 ? 'border-b border-slate-50' : ''}
                      ${isSelected ? 'bg-blue-50' : ''}
                    `}
                  >
                    <div className={`p-1.5 rounded-lg ${isSelected ? 'bg-blue-500' : 'bg-slate-100'}`}>
                      <FolderOpen size={14} className={isSelected ? 'text-white' : 'text-slate-400'} />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className={`truncate font-medium text-sm ${isSelected ? 'text-blue-700' : 'text-slate-700'}`}>
                        {displayName}
                      </div>
                      {service !== displayName && (
                        <div className="text-xs text-slate-400 truncate">
                          {service}
                        </div>
                      )}
                    </div>
                    {isSelected && (
                      <Check size={16} className="text-blue-600 flex-shrink-0" />
                    )}
                  </button>
                );
              })
            )}
          </div>

          {/* Footer */}
          {services.length > 10 && (
            <div className="p-3 border-t border-slate-100 bg-gradient-to-r from-slate-50 to-white">
              <p className="text-xs text-slate-500 text-center">
                💡 Use search to quickly find a service
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
