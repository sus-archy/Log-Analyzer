"""
Ingest service - handles log ingestion from files and API.
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from ..core.config import settings
from ..core.logging import get_logger
from ..core.time import now_utc_iso
from ..parsers.folder_loader import (
    discover_log_files,
    infer_service_name_from_path,
    read_file_lines,
    get_file_format,
)
from ..parsers.jsonl_parser import parse_jsonl_line
from ..parsers.text_parser import parse_text_line, parse_csv_structured_line
from ..parsers.normalize import normalize_event
from ..parsers.drain_miner import mine_template
from ..schemas.ingest import IngestStats
from ..schemas.logs import NormalizedLogEvent
from ..storage.db import get_db
from ..storage.templates_repo import TemplatesRepo
from ..storage.logs_repo import LogsRepo
from ..storage.ingested_files_repo import IngestedFilesRepo, compute_file_hash

logger = get_logger(__name__)


class IngestService:
    """Service for ingesting log data."""
    
    def __init__(self):
        self._template_cache: Set[Tuple[str, str, int]] = set()
    
    async def ingest_from_folder(
        self,
        folder_path: Optional[Path] = None,
        tenant_id: Optional[str] = None,
        skip_duplicates: bool = True,
    ) -> IngestStats:
        """
        Ingest all log files from a folder.
        
        Args:
            folder_path: Folder to ingest from (defaults to config)
            tenant_id: Tenant ID override
            skip_duplicates: If True, skip files that have already been ingested
            
        Returns:
            Ingestion statistics
        """
        if folder_path is None:
            folder_path = settings.logs_folder_resolved
        
        if tenant_id is None:
            tenant_id = settings.tenant_id_default
        
        logger.info(f"Starting ingestion from {folder_path}")
        
        stats = IngestStats(
            files_processed=0,
            files_skipped=0,
            lines_processed=0,
            events_inserted=0,
            templates_discovered=0,
            errors=[],
            skipped_files=[],
        )
        
        # Discover files
        files = discover_log_files(folder_path)
        
        if not files:
            logger.warning(f"No log files found in {folder_path}")
            return stats
        
        total_files = len(files)
        print(f"Found {total_files} files to process")
        
        db = await get_db()
        templates_repo = TemplatesRepo(db)
        logs_repo = LogsRepo(db)
        ingested_files_repo = IngestedFilesRepo(db)
        
        for i, filepath in enumerate(files, 1):
            print(f"[{i}/{total_files}] Processing: {filepath.name}...", end=" ", flush=True)
            
            # Check for duplicate files
            if skip_duplicates:
                try:
                    file_hash = compute_file_hash(filepath)
                    existing = await ingested_files_repo.get_ingested_file(file_hash)
                    
                    if existing:
                        stats.files_skipped += 1
                        stats.skipped_files.append(filepath.name)
                        print(f"SKIPPED (duplicate of {existing['file_name']})")
                        continue
                except Exception as e:
                    logger.warning(f"Could not compute hash for {filepath}: {e}")
                    file_hash = None
            else:
                file_hash = None
            
            try:
                file_stats = await self._ingest_file(
                    filepath=filepath,
                    base_folder=folder_path,
                    tenant_id=tenant_id,
                    templates_repo=templates_repo,
                    logs_repo=logs_repo,
                )
                
                stats.files_processed += 1
                stats.lines_processed += file_stats["lines"]
                stats.events_inserted += file_stats["events"]
                stats.templates_discovered += file_stats["templates"]
                print(f"{file_stats['events']} events, {file_stats['templates']} templates")
                
                # Record that we've ingested this file
                if file_hash:
                    try:
                        await ingested_files_repo.record_ingested_file(
                            file_path=filepath,
                            file_hash=file_hash,
                            lines_processed=file_stats["lines"],
                            events_inserted=file_stats["events"],
                            templates_discovered=file_stats["templates"],
                            tenant_id=tenant_id,
                        )
                    except Exception as e:
                        logger.warning(f"Could not record file {filepath}: {e}")
                
            except Exception as e:
                error_msg = f"Error processing {filepath}: {e}"
                logger.error(error_msg)
                stats.errors.append(error_msg)
                print(f"ERROR: {e}")
        
        await db.commit()
        logger.info(
            f"Ingestion complete: {stats.files_processed} files, "
            f"{stats.events_inserted} events, {stats.templates_discovered} templates"
        )
        
        return stats
    
    async def ingest_single_file(
        self,
        file_path: Path,
        service_name: Optional[str] = None,
        tenant_id: Optional[str] = None,
        skip_duplicates: bool = True,
    ) -> IngestStats:
        """
        Ingest a single log file (e.g., from upload).
        
        Args:
            file_path: Path to the file
            service_name: Service name override (derived from filename if not provided)
            tenant_id: Tenant ID override
            skip_duplicates: If True, skip files that have already been ingested
            
        Returns:
            Ingestion statistics
        """
        if tenant_id is None:
            tenant_id = settings.tenant_id_default
        
        if service_name is None:
            service_name = file_path.stem.replace("_", "-")
        
        logger.info(f"Ingesting single file: {file_path} as service={service_name}")
        
        stats = IngestStats(
            files_processed=0,
            files_skipped=0,
            lines_processed=0,
            events_inserted=0,
            templates_discovered=0,
            errors=[],
            skipped_files=[],
        )
        
        db = await get_db()
        templates_repo = TemplatesRepo(db)
        logs_repo = LogsRepo(db)
        ingested_files_repo = IngestedFilesRepo(db)
        
        # Check for duplicate files
        file_hash = None
        if skip_duplicates:
            try:
                file_hash = compute_file_hash(file_path)
                existing = await ingested_files_repo.get_ingested_file(file_hash)
                
                if existing:
                    stats.files_skipped = 1
                    stats.skipped_files.append(file_path.name)
                    logger.info(f"Skipping duplicate file: {file_path.name} (duplicate of {existing['file_name']})")
                    return stats
            except Exception as e:
                logger.warning(f"Could not compute hash for {file_path}: {e}")
        
        try:
            file_stats = await self._ingest_file(
                filepath=file_path,
                base_folder=file_path.parent,
                tenant_id=tenant_id,
                templates_repo=templates_repo,
                logs_repo=logs_repo,
            )
            
            stats.files_processed = 1
            stats.lines_processed = file_stats["lines"]
            stats.events_inserted = file_stats["events"]
            stats.templates_discovered = file_stats["templates"]
            
            # Record that we've ingested this file
            if file_hash:
                try:
                    await ingested_files_repo.record_ingested_file(
                        file_path=file_path,
                        file_hash=file_hash,
                        lines_processed=file_stats["lines"],
                        events_inserted=file_stats["events"],
                        templates_discovered=file_stats["templates"],
                        tenant_id=tenant_id,
                    )
                except Exception as e:
                    logger.warning(f"Could not record file {file_path}: {e}")
            
        except Exception as e:
            error_msg = f"Error processing {file_path}: {e}"
            logger.error(error_msg)
            stats.errors.append(error_msg)
            raise
        
        await db.commit()
        logger.info(f"Single file ingestion complete: {stats.events_inserted} events")
        
        return stats
    
    async def _ingest_file(
        self,
        filepath: Path,
        base_folder: Path,
        tenant_id: str,
        templates_repo: TemplatesRepo,
        logs_repo: LogsRepo,
    ) -> Dict[str, int]:
        """Ingest a single file."""
        logger.debug(f"Processing file: {filepath}")
        
        default_service = infer_service_name_from_path(filepath, base_folder)
        file_format = get_file_format(filepath)
        
        lines_count = 0
        events_count = 0
        templates_count = 0
        
        events_batch: List[NormalizedLogEvent] = []
        batch_size = 1000  # Larger batch for better performance
        
        if file_format == "csv":
            # Handle CSV files specially
            async for event, is_new_template in self._parse_csv_file(
                filepath, default_service, tenant_id, templates_repo
            ):
                events_batch.append(event)
                lines_count += 1
                
                if is_new_template:
                    templates_count += 1
                
                if len(events_batch) >= batch_size:
                    await logs_repo.insert_logs_batch(events_batch)
                    events_count += len(events_batch)
                    events_batch = []
        else:
            # Handle text and jsonl files
            for line_num, line in read_file_lines(filepath):
                lines_count += 1
                
                # Parse line based on format
                if file_format == "jsonl":
                    parsed = parse_jsonl_line(line, default_service)
                    if parsed is None:
                        continue
                else:
                    parsed = parse_text_line(line, default_service)
                
                # Mine template
                message = parsed.get("message", line)
                service_name = parsed.get("service_name", default_service)
                
                template_text, parameters, template_hash = mine_template(
                    message, service_name
                )
                
                # Normalize event
                event = normalize_event(
                    parsed,
                    tenant_id=tenant_id,
                    template_hash=template_hash,
                    parameters=parameters,
                )
                
                # Handle template
                is_new = await self._handle_template(
                    tenant_id=tenant_id,
                    service_name=service_name,
                    template_hash=template_hash,
                    template_text=template_text,
                    timestamp_utc=event.timestamp_utc,
                    templates_repo=templates_repo,
                )
                
                if is_new:
                    templates_count += 1
                
                events_batch.append(event)
                
                if len(events_batch) >= batch_size:
                    await logs_repo.insert_logs_batch(events_batch)
                    events_count += len(events_batch)
                    events_batch = []
        
        # Insert remaining events
        if events_batch:
            await logs_repo.insert_logs_batch(events_batch)
            events_count += len(events_batch)
        
        return {
            "lines": lines_count,
            "events": events_count,
            "templates": templates_count,
        }
    
    async def _parse_csv_file(
        self,
        filepath: Path,
        default_service: str,
        tenant_id: str,
        templates_repo: TemplatesRepo,
    ):
        """Parse CSV file and yield events."""
        encodings = ["utf-8", "latin-1", "cp1252"]
        
        for encoding in encodings:
            try:
                with open(filepath, "r", encoding=encoding, errors="replace") as f:
                    reader = csv.DictReader(f)
                    
                    for row in reader:
                        parsed = parse_csv_structured_line(row, default_service)
                        
                        message = parsed.get("message", "")
                        if not message:
                            continue
                        
                        service_name = parsed.get("service_name", default_service)
                        
                        # Mine template
                        template_text, parameters, template_hash = mine_template(
                            message, service_name
                        )
                        
                        # Normalize event
                        event = normalize_event(
                            parsed,
                            tenant_id=tenant_id,
                            template_hash=template_hash,
                            parameters=parameters,
                        )
                        
                        # Handle template
                        is_new = await self._handle_template(
                            tenant_id=tenant_id,
                            service_name=service_name,
                            template_hash=template_hash,
                            template_text=template_text,
                            timestamp_utc=event.timestamp_utc,
                            templates_repo=templates_repo,
                        )
                        
                        yield event, is_new
                return
            except UnicodeDecodeError:
                continue
    
    async def _handle_template(
        self,
        tenant_id: str,
        service_name: str,
        template_hash: int,
        template_text: str,
        timestamp_utc: str,
        templates_repo: TemplatesRepo,
    ) -> bool:
        """
        Handle template caching and storage.
        
        Returns:
            True if new template was discovered
        """
        cache_key = (tenant_id, service_name, template_hash)
        
        # Check cache first
        if cache_key in self._template_cache:
            return False
        
        # Add to cache
        self._template_cache.add(cache_key)
        
        # Upsert to database
        is_new = await templates_repo.upsert_template(
            tenant_id=tenant_id,
            service_name=service_name,
            template_hash=template_hash,
            template_text=template_text,
            timestamp_utc=timestamp_utc,
        )
        
        return is_new
    
    async def ingest_events(
        self,
        events: List[dict],
        tenant_id: Optional[str] = None,
    ) -> IngestStats:
        """
        Ingest events from API.
        
        Args:
            events: List of raw event dicts
            tenant_id: Tenant ID override
            
        Returns:
            Ingestion statistics
        """
        if tenant_id is None:
            tenant_id = settings.tenant_id_default
        
        stats = IngestStats(
            files_processed=0,
            lines_processed=len(events),
            events_inserted=0,
            templates_discovered=0,
            errors=[],
        )
        
        db = await get_db()
        templates_repo = TemplatesRepo(db)
        logs_repo = LogsRepo(db)
        
        normalized_events: List[NormalizedLogEvent] = []
        
        for raw_event in events:
            try:
                message = raw_event.get("message", "")
                service_name = raw_event.get("service_name", "unknown")
                
                # Mine template
                template_text, parameters, template_hash = mine_template(
                    message, service_name
                )
                
                # Normalize
                event = normalize_event(
                    raw_event,
                    tenant_id=tenant_id,
                    template_hash=template_hash,
                    parameters=parameters,
                )
                
                # Handle template
                is_new = await self._handle_template(
                    tenant_id=tenant_id,
                    service_name=service_name,
                    template_hash=template_hash,
                    template_text=template_text,
                    timestamp_utc=event.timestamp_utc,
                    templates_repo=templates_repo,
                )
                
                if is_new:
                    stats.templates_discovered += 1
                
                normalized_events.append(event)
                
            except Exception as e:
                error_msg = f"Error processing event: {e}"
                logger.error(error_msg)
                stats.errors.append(error_msg)
        
        # Insert all events
        if normalized_events:
            await logs_repo.insert_logs_batch(normalized_events)
            stats.events_inserted = len(normalized_events)
        
        await db.commit()
        
        return stats


# Global service instance
_ingest_service: Optional[IngestService] = None


def get_ingest_service() -> IngestService:
    """Get global ingest service instance."""
    global _ingest_service
    if _ingest_service is None:
        _ingest_service = IngestService()
    return _ingest_service
