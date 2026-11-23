"""Non-training R03 verification; read-only and independent of train/test split."""
import argparse, ast, pathlib, subprocess, sys
from hashlib import sha256

parser=argparse.ArgumentParser()
parser.add_argument('--project',default='.')
args=parser.parse_args()
root=pathlib.Path(args.project).resolve()
app=root/'app.py'
css=root/'ui'/'uas_theme.css'
helper=root/'ui'/'r03_helpers.py'
for path in (app,css,helper):
    if not path.exists(): raise SystemExit('FAIL: missing '+str(path))
txt=app.read_text(encoding='utf8')
ast.parse(txt)
for x in ('# R03_SAFE_PATCH_APPLIED','guard_html(','invalidated_html()', '.change(fn=_r03_clear_results'):
    if x not in txt: raise SystemExit('FAIL: missing code marker '+x)
print('R03 STATIC GUARD: PASS')
print('APP SHA256:',sha256(app.read_bytes()).hexdigest())
print('MODEL FILES UNTOUCHED BY R03 PATCHER: design guarantee; run original SHA256 verifier separately')
print('NEXT: python scripts/13_verify_app.py --check')
