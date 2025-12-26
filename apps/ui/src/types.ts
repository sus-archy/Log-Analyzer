/**
 * TypeScript types for LogMind AI
 */

export interface LogEvent {
  id: number;
  tenant_id: string;
  service_name: string;
  environment: string;
  timestamp_utc: string;
  severity: number;
  severity_name: string;
  host: string;
  template_hash: string;  // String to handle 64-bit integers
  template_text: string | null;
  parameters: any[];
  trace_id: string;
  span_id: string;
  attributes: Record<string, any>;
  body_raw: string;
}

export interface LogQueryResponse {
  logs: LogEvent[];
  total: number;
  limit: number;
  offset: number;
  has_more: boolean;
}

export interface Template {
  template_hash: string;  // String to handle 64-bit integers
  template_text: string;
  count: number;
  sample_log_id: number | null;
  first_seen_utc: string;
  last_seen_utc: string;
}

export interface TopTemplatesResponse {
  templates: Template[];
  service_name: string;
  from_time: string;
  to_time: string;
  total: number;
}

export interface TemplateOccurrence {
  log_id: number;
  timestamp_utc: string;
  parameters: string[];
  severity: number;
  host: string;
}

export interface TemplateDetail {
  template_hash: string;  // String to handle 64-bit integers
  template_text: string;
  service_name: string;
  first_seen_utc: string;
  last_seen_utc: string;
  total_count: number;
  occurrences: TemplateOccurrence[];
  embedding_state: string;
}

export interface SemanticSearchResult {
  template_hash: string;  // String to handle 64-bit integers
  template_text: string;
  score: number;
  count: number;
  sample_log_ids: number[];
}

export interface SemanticSearchResponse {
  results: SemanticSearchResult[];
  query: string;
  service_name: string;
  from_time: string;
  to_time: string;
}

export interface Citation {
  type: 'template' | 'log';
  service_name?: string;
  template_hash?: number;
  template_text?: string;
  log_id?: number;
  relevance?: string;
}

export interface ChatResponse {
  answer: string;
  citations: Citation[];
  confidence: string;
  next_steps: string[];
  metadata: Record<string, any>;
}

export interface IngestStats {
  files_processed: number;
  lines_processed: number;
  events_inserted: number;
  templates_discovered: number;
  errors: string[];
}

export interface IngestResponse {
  success: boolean;
  stats: IngestStats;
  message: string;
}

export interface ServicesResponse {
  services: string[];
}

// Severity helpers
export const SEVERITY_COLORS: Record<number, string> = {
  0: 'text-slate-400',    // TRACE
  1: 'text-slate-500',    // DEBUG
  2: 'text-sky-500',      // INFO
  3: 'text-amber-500',    // WARN
  4: 'text-rose-500',     // ERROR
  5: 'text-fuchsia-600',  // FATAL
};

export const SEVERITY_BG_COLORS: Record<number, string> = {
  0: '',
  1: '',
  2: '',
  3: 'bg-amber-50',
  4: 'bg-rose-50',
  5: 'bg-fuchsia-50',
};

export const SEVERITY_NAMES: Record<number, string> = {
  0: 'TRACE',
  1: 'DEBUG',
  2: 'INFO',
  3: 'WARN',
  4: 'ERROR',
  5: 'FATAL',
};
