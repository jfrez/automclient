# -*- mode: python ; coding: utf-8 -*-


block_cipher = None

a = Analysis(
    ['..\\main.py'],
    pathex=[],
    binaries=[],
    datas=[('../images/*', 'images/')],
    hiddenimports=[
        'engineio.async_eventlet',
        'openpyxl.cell._writer',
        'notebook.services.shutdown',
        'engineio.async_drivers.aiohttp',
        'engineio.async_aiohttp',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Autom',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    windowed=True,
    disable_windowed_traceback=True,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['logo.ico'],
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Autom',
    )

# Script para mover dependencias a la carpeta "libraries"
import shutil
import os

dist_dir = os.path.join('dist', 'Autom')
libs_dir = os.path.join(dist_dir, 'libraries')

if not os.path.exists(libs_dir):
    os.makedirs(libs_dir)

for item in os.listdir(dist_dir):
    item_path = os.path.join(dist_dir, item)
    if os.path.isfile(item_path) and item != 'Autom.exe':
        shutil.move(item_path, libs_dir)

