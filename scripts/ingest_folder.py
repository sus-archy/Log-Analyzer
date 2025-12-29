#!/usr/bin/env python3
"""
Ingest logs from a folder - SCALABLE for files of ANY size.

Features:
- Streaming: Never loads full file in memory
- Incremental saves: Commits every 2000 events (crash-safe)
- Fast mode: Auto-detects large files and uses faster processing
- Progress: Real-time line-by-line progress for large files

Usage: python scripts/ingest_folder.py [path]
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Add the api app to path
sys.path.insert(0, str(Path(__file__).parent.parent / "apps" / "api"))

from app.core.config import settings
from app.storage.db import init_db, close_db
from app.services.ingest_service import IngestService


class SimpleProgress:
    """Simple progress tracker without external dependencies."""
    
    def __init__(self):
        self.start_time = time.time()
        self.files_done = 0
        self.total_files = 0
    
    def update(self, current: int, total: int, message: str, metadata: Dict[str, Any]):
        phase = metadata.get("phase", "")
        
        if phase == "discovery":
            self.total_files = total
            
        elif phase == "ingesting":
            file_name = metadata.get("file", "")
            events = metadata.get("events_so_far", 0)
            elapsed = time.time() - self.start_time
            rate = events / elapsed if elapsed > 0 else 0
            
            print(f"[{current}/{total}] 📁 {file_name} | Total: {events:,} events | {rate:.0f}/sec")
            
        elif phase == "complete":
            elapsed = time.time() - self.start_time
            total_events = metadata.get("total_events", 0)
            print(f"\n✅ All files processed in {elapsed:.1f}s")


async def main():
    if len(sys.argv) > 1:
        folder = Path(sys.argv[1])
    else:
        folder = settings.logs_folder_resolved
    
    print("=" * 70)
    print("🚀 LOG INGESTION - Scalable Streaming Mode")
    print("=" * 70)
    print(f"📂 Source:    {folder}")
    print(f"💾 Database:  {settings.db_path_resolved}")
    print(f"⚡ Features:  Streaming | Incremental saves | Fast mode for large files")
    print("=" * 70)
    print()
    
    await init_db()
    
    tracker = SimpleProgress()
    ingest_service = IngestService()
    
    start = time.time()
    result = await ingest_service.ingest_from_folder(
        folder,
        progress_callback=tracker.update
    )
    elapsed = time.time() - start
    
    print()
    print("=" * 70)
    print("📊 INGESTION COMPLETE")
    print("=" * 70)
    print(f"✅ Files processed:    {result.files_processed:>12}")
    print(f"⏭️  Files skipped:      {result.files_skipped:>12}")
    print(f"📝 Lines processed:    {result.lines_processed:>12,}")
    print(f"📦 Events inserted:    {result.events_inserted:>12,}")
    print(f"🔖 Templates found:    {result.templates_discovered:>12}")
    if elapsed > 0:
        print(f"⚡ Speed:              {result.events_inserted/elapsed:>12.0f} events/sec")
    print(f"⏱️  Total time:         {elapsed:>12.1f}s")
    print("=" * 70)
    
    if result.errors:
        print(f"\n⚠️  {len(result.errors)} errors:")
        for e in result.errors[:5]:
            print(f"   • {e[:80]}")
        if len(result.errors) > 5:
            print(f"   ... and {len(result.errors) - 5} more")
    
    await close_db()


if __name__ == "__main__":
    asyncio.run(main())
