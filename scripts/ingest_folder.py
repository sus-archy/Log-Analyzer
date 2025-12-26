#!/usr/bin/env python3
"""
Ingest logs from a specific folder or file.
Usage: python scripts/ingest_folder.py [path]
"""

import asyncio
import sys
from pathlib import Path

# Add the api app to path
sys.path.insert(0, str(Path(__file__).parent.parent / "apps" / "api"))

from app.core.config import settings
from app.storage.db import init_db, close_db
from app.services.ingest_service import IngestService


async def main():
    # Get path from args or use default
    if len(sys.argv) > 1:
        folder = Path(sys.argv[1])
    else:
        folder = settings.logs_folder_resolved
    
    print(f"Ingesting logs from: {folder}")
    print(f"Database: {settings.db_path_resolved}")
    
    # Initialize database
    await init_db()
    
    # Create ingest service
    ingest_service = IngestService()
    
    # Run ingest
    result = await ingest_service.ingest_from_folder(folder)
    
    print(f"\n=== Ingest Complete ===")
    print(f"Files processed: {result.files_processed}")
    print(f"Lines processed: {result.lines_processed}")
    print(f"Events inserted: {result.events_inserted}")
    print(f"Templates discovered: {result.templates_discovered}")
    print(f"Errors: {len(result.errors)}")
    
    if result.errors:
        print("\nErrors:")
        for error in result.errors[:10]:
            print(f"  - {error}")
        if len(result.errors) > 10:
            print(f"  ... and {len(result.errors) - 10} more")
    
    await close_db()


if __name__ == "__main__":
    asyncio.run(main())
