# -*- mode: python ; coding: utf-8 -*-
block_cipher = None

a = Analysis(
    ['ae_gui_qt.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('dia_thresholds.json', '.'),
        ('gui_config.json', '.'),
        ('output_xgb/xgb_break_tool.pkl', 'output_xgb'),
    ],
    hiddenimports=['PyQt5.sip'],
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
    name='断刀检测系统',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='断刀检测系统',
)
