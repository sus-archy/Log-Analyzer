#!/usr/bin/env python3
"""Deep scan of LogMind project for issues and conflicts."""

import os
import sys
import glob
import sqlite3

# Change to project root
os.chdir('/home/bug/Desktop/Log_Analyzer')

print('=' * 60)
print('PROJECT DEEP SCAN')
print('=' * 60)

errors = []
warnings = []

# 1. Check Python imports
print('\n[1] CHECKING PYTHON IMPORTS...')
import_ok = []
try:
    sys.path.insert(0, 'apps/api')
    from app.main import app
    import_ok.append('app.main')
    from app.ml import initialize_models
    import_ok.append('app.ml')
    from app.storage import get_db, init_db
    import_ok.append('app.storage')
    from app.services import get_embedding_service
    import_ok.append('app.services')
    from app.vector import get_faiss_index
    import_ok.append('app.vector')
    print(f'  ✅ All core imports OK: {import_ok}')
except Exception as e:
    errors.append(f'Import error: {e}')
    print(f'  ❌ Import error: {e}')

# 2. Check required files
print('\n[2] CHECKING REQUIRED FILES...')
required_files = [
    'data/logmind.sqlite',
    'data/models/anomaly_detector.pkl',
    'data/models/log_classifier.pkl',
    'data/models/security_detector.pkl',
    'apps/api/app/main.py',
    'apps/api/requirements.txt',
    'apps/ui/package.json',
]
for f in required_files:
    if os.path.exists(f):
        size = os.path.getsize(f)
        print(f'  ✅ {f} ({size:,} bytes)')
    else:
        errors.append(f'Missing: {f}')
        print(f'  ❌ MISSING: {f}')

# 3. Check for potential conflicts
print('\n[3] CHECKING FOR CONFLICTS...')
model_files = glob.glob('**/*.pkl', recursive=True)
print(f'  Model files found: {model_files}')

pyc_files = glob.glob('**/__pycache__/**/*.pyc', recursive=True)
print(f'  Pycache files: {len(pyc_files)}')

config_files = glob.glob('**/pyrightconfig.json', recursive=True)
print(f'  Pyright configs: {config_files}')

# Check for FAISS index
if os.path.exists('data/faiss.index'):
    print(f'  ✅ FAISS index exists')
else:
    warnings.append('FAISS index missing - embeddings need to be processed')
    print(f'  ⚠️  FAISS index missing (not critical, will be built when embeddings are processed)')

# 4. Check database consistency
print('\n[4] DATABASE CONSISTENCY...')
conn = sqlite3.connect('data/logmind.sqlite')
cursor = conn.cursor()

# Check logs
cursor.execute('SELECT COUNT(*) FROM logs_stream')
log_count = cursor.fetchone()[0]
print(f'  Total logs: {log_count:,}')

# Check templates
cursor.execute('SELECT COUNT(*) FROM log_templates')
template_count = cursor.fetchone()[0]
print(f'  Total templates: {template_count:,}')

# Check for orphan templates
cursor.execute('SELECT COUNT(*) FROM log_templates WHERE template_hash NOT IN (SELECT DISTINCT template_hash FROM logs_stream WHERE template_hash IS NOT NULL)')
orphan = cursor.fetchone()[0]
if orphan > 0:
    warnings.append(f'Orphan templates (no logs): {orphan}')
print(f'  Orphan templates (no logs): {orphan}')

# Check templates with vectors
cursor.execute("SELECT COUNT(*) FROM log_templates WHERE embedding_state = 'ready'")
ready = cursor.fetchone()[0]
cursor.execute('SELECT COUNT(*) FROM template_vectors')
vectors = cursor.fetchone()[0]
print(f'  Templates with ready state: {ready}')
print(f'  Vectors in table: {vectors}')
if ready > 0 and vectors == 0:
    errors.append('INCONSISTENCY: Templates marked ready but no vectors exist')
    print(f'  ❌ INCONSISTENCY: Templates marked ready but no vectors!')

# Check embedding states distribution
print('\n  Embedding states:')
cursor.execute('SELECT embedding_state, COUNT(*) FROM log_templates GROUP BY embedding_state')
for row in cursor.fetchall():
    state = row[0] or 'none'
    print(f'    - {state}: {row[1]:,}')

# Check services
print('\n  Top services by log count:')
cursor.execute('SELECT service_name, COUNT(*) FROM logs_stream GROUP BY service_name ORDER BY COUNT(*) DESC LIMIT 5')
for row in cursor.fetchall():
    print(f'    - {row[0]}: {row[1]:,}')

conn.close()

# 5. Check ML models can be loaded
print('\n[5] ML MODEL VALIDATION...')
try:
    import pickle
    for model_name in ['anomaly_detector', 'log_classifier', 'security_detector']:
        path = f'data/models/{model_name}.pkl'
        with open(path, 'rb') as f:
            data = pickle.load(f)
        print(f'  ✅ {model_name}: keys={list(data.keys())[:3]}...')
except Exception as e:
    errors.append(f'Model load error: {e}')
    print(f'  ❌ Model error: {e}')

# Summary
print('\n' + '=' * 60)
print('SCAN SUMMARY')
print('=' * 60)

if errors:
    print(f'\n❌ ERRORS ({len(errors)}):')
    for e in errors:
        print(f'   - {e}')

if warnings:
    print(f'\n⚠️  WARNINGS ({len(warnings)}):')
    for w in warnings:
        print(f'   - {w}')

if not errors and not warnings:
    print('\n✅ NO ISSUES FOUND - Project is healthy!')
elif not errors:
    print('\n✅ No critical errors - minor warnings only')
else:
    print(f'\n❌ {len(errors)} error(s) need attention')

print('=' * 60)
