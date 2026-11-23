"""Conservative R03 updater for an *existing*, locally installed R02 Gradio app.

- --inspect / --dry-run never modify project
- --apply adds event invalidation, HTML sanitization wrapper, CSS addon
- --rollback restores newest backup made by this script
- Abort if Gradio component declarations/event wiring cannot be identified safely.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
SIGN = '# R03_SAFE_PATCH_APPLIED'


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _names(expr, definitions=None):
    if expr is None:
        return []
    if isinstance(expr, ast.Name):
        if definitions and expr.id in definitions and isinstance(definitions[expr.id], (ast.List, ast.Tuple)):
            return _names(definitions[expr.id], definitions)
        return [expr.id]
    if isinstance(expr, (ast.List, ast.Tuple)):
        answer = []
        for el in expr.elts:
            answer.extend(_names(el, definitions))
        return answer
    return []


def _kw(node, name, position):
    for k in node.keywords:
        if k.arg == name:
            return k.value
    return node.args[position] if len(node.args) > position else None


def _component_type(call):
    if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def analyze(source):
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise ValueError('app.py tidak dapat di-parse: ' + str(e))
    defs = {}
    definitions = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    defs[target.id] = _component_type(node.value)
                    definitions[target.id] = node.value
    candidates=[]
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute) or node.func.attr != 'click':
            continue
        inputs = _names(_kw(node, 'inputs', 1), definitions)
        outputs = _names(_kw(node, 'outputs', 2), definitions)
        if not inputs or not outputs:
            continue
        counts = [defs.get(name) for name in inputs]
        if 'Textbox' not in counts or 'Radio' not in counts:
            continue
        output_types=[defs.get(name) for name in outputs]
        if not all(t in ('HTML','Markdown','Textbox') for t in output_types):
            continue
        func=_kw(node,'fn',0)
        if func is None or not isinstance(func,(ast.Name,ast.Attribute)):
            continue
        # Find the enclosing event statement to inject change handlers after the click call.
        stmt = None
        for top in ast.walk(tree):
            if isinstance(top,(ast.Expr, ast.Assign, ast.AnnAssign)) and top.lineno <= node.lineno <= top.end_lineno:
                if stmt is None or (top.end_lineno-top.lineno)<(stmt.end_lineno-stmt.lineno):
                    stmt=top
        if stmt is None or stmt.lineno > node.lineno:
            continue
        candidates.append((node,stmt,inputs,outputs,output_types,func))
    if len(candidates)!=1:
        raise ValueError('Ditemukan %d handler Analisis yang cocok, diharapkan tepat 1. Patch dihentikan agar aman.' % len(candidates))
    node,stmt,inputs,outputs,types,fn=candidates[0]
    listeners=[x for x in inputs if defs.get(x) in ('Textbox','Radio','Checkbox')]
    if not any(defs.get(x)=='Textbox' for x in listeners) or not any(defs.get(x)=='Radio' for x in listeners):
        raise ValueError('Textbox dan Radio input tidak dikenali dengan aman.')
    if len(set(outputs))!=len(outputs):
        raise ValueError('Output HTML duplikat, patch dihentikan.')
    html_positions=tuple(i for i,t in enumerate(types) if t=='HTML')
    if not html_positions:
        raise ValueError('Tidak ditemukan output gr.HTML untuk diperiksa dan diamankan.')
    return {
        'tree':tree, 'click':node,'statement':stmt,'inputs':inputs,'outputs':outputs,
        'output_types':types,'fn':fn,'html_positions':html_positions,
        'listeners':list(dict.fromkeys(listeners))
    }


def to_source_patch(source, info):
    if SIGN in source:
        raise ValueError('app.py sudah memiliki patch R03. Tidak akan memasang dua kali.')
    lines=source.splitlines(keepends=True)
    node=info['click']
    fn=info['fn']
    # Replace only the original fn expression. No model inference code is changed.
    fn_line=lines[fn.lineno-1]
    if fn.end_lineno!=fn.lineno:
        raise ValueError('Ekspresi callback terlalu kompleks. Hentikan patch dan audit manual.')
    start,end=fn.col_offset,fn.end_col_offset
    original_fn=fn_line[start:end]
    if not original_fn or '\n' in original_fn:
        raise ValueError('Tidak dapat membaca nama callback.')
    positions=', '.join(str(i) for i in info['html_positions'])
    new_expr='guard_html('+original_fn+', html_positions=('+positions+(',' if len(info['html_positions'])==1 else '')+'))'
    lines[fn.lineno-1]=fn_line[:start]+new_expr+fn_line[end:]
    # Invalidation event is attached to input changes (including example-button changes), mode, and checkbox.
    stmt=info['statement']
    indent=' '*(stmt.col_offset)
    cancel_clause=''
    if isinstance(stmt, ast.Expr) and stmt.value is node:
        # Also cancel pending analysis jobs when Gradio exposes a direct click dependency.
        first_idx=stmt.lineno-1
        lines[first_idx]=lines[first_idx][:stmt.col_offset]+'_r03_analysis_event = '+lines[first_idx][stmt.col_offset:]
        cancel_clause=', cancels=[_r03_analysis_event]'
    updates=[]
    for index,t in enumerate(info['output_types']):
        if index==0:
            updates.append('gr.update(value=invalidated_html())')
        else:
            updates.append('gr.update(value="")')
    returns = updates[0] if len(updates)==1 else '('+', '.join(updates)+')'
    injection=[
       '\n',
       indent+SIGN+'\n',
       indent+'def _r03_clear_results():\n',
       indent+'    return '+returns+'\n',
    ]
    for var in info['listeners']:
        injection.append(indent+var+'.change(fn=_r03_clear_results, outputs=['+', '.join(info['outputs'])+'], queue=False, show_progress="hidden"'+cancel_clause+')\n')
    # Avoid altering the R02 existing markup. Add callbacks after its existing click binding.
    lines[stmt.end_lineno:stmt.end_lineno]=injection
    # Insert import after module docstring / __future__ imports / regular imports.
    new=''.join(lines)
    parsed=ast.parse(new)
    insert=0
    for stmt0 in parsed.body:
        if isinstance(stmt0,(ast.Import,ast.ImportFrom)) and stmt0.lineno < info['click'].lineno:
            insert=max(insert,stmt0.end_lineno)
    new_lines=new.splitlines(keepends=True)
    new_lines[insert:insert]=['from ui.r03_helpers import guard_html, invalidated_html\n']
    final=''.join(new_lines)
    ast.parse(final)
    return final


def report(project, info, status, warning=None):
    return dict(status=status, project=str(project),
                inputs=info.get('inputs',[]) if info else [],
                outputs=info.get('outputs',[]) if info else [],
                listeners=info.get('listeners',[]) if info else [],
                output_types=info.get('output_types',[]) if info else [],
                warning=warning)


def main():
    parser=argparse.ArgumentParser(description='Safe, project-aware R03 Gradio UI patcher')
    parser.add_argument('--project',default='.',help='Folder proyek SMS-Spam-Indonesia')
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--inspect',action='store_true',help='Periksa kecocokan, tidak mengubah apa pun')
    group.add_argument('--dry-run',action='store_true',help='Cetak preview diff, tanpa perubahan')
    group.add_argument('--apply',action='store_true',help='Backup lalu terapkan patch')
    group.add_argument('--rollback',action='store_true',help='Pulihkan backup patch R03 terbaru')
    args=parser.parse_args()
    project=Path(args.project).resolve()
    app=project/'app.py'
    css=project/'ui'/'uas_theme.css'
    helper_dest=project/'ui'/'r03_helpers.py'
    backup_root=project/'outputs'/'reports'/'r03_backups'
    if args.rollback:
        backups=sorted((x for x in backup_root.glob('*') if x.is_dir()),reverse=True) if backup_root.exists() else []
        if not backups: raise SystemExit('Tidak ada backup R03 yang ditemukan. Tidak ada perubahan.')
        backup=backups[0]
        if not (backup/'app.py').exists(): raise SystemExit('Backup tidak lengkap; rollback ditolak.')
        shutil.copy2(backup/'app.py',app)
        if (backup/'uas_theme.css').exists():
            css.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(backup/'uas_theme.css',css)
        if (backup/'r03_helpers.py').exists():
            shutil.copy2(backup/'r03_helpers.py',helper_dest)
        elif helper_dest.exists():
            helper_dest.unlink()
        print('ROLLBACK_R03_PASS | backup:',backup)
        return
    if not app.is_file(): raise SystemExit('app.py tidak ada: '+str(app))
    source=app.read_text(encoding='utf-8-sig')
    try:
        info=analyze(source)
        modified=to_source_patch(source,info)
    except Exception as exc:
        print('R03_PREFLIGHT_BLOCKED:',exc)
        print('TIDAK ADA FILE YANG DIUBAH. Kirim app.py R02 untuk patch manual yang presisi.')
        return 2
    summary=report(project, info, 'R03_PREFLIGHT_PASS')
    print(json.dumps(summary,indent=2,ensure_ascii=False))
    if args.inspect: return
    if args.dry_run:
        print(''.join(difflib.unified_diff(source.splitlines(True),modified.splitlines(True),fromfile='R02/app.py',tofile='R03/app.py')))
        return
    # All checks before writes; never touch model files or original training scripts.
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    backup=backup_root/stamp
    backup.mkdir(parents=True,exist_ok=False)
    shutil.copy2(app,backup/'app.py')
    if css.exists(): shutil.copy2(css,backup/'uas_theme.css')
    if helper_dest.exists(): shutil.copy2(helper_dest,backup/'r03_helpers.py')
    extra=(ROOT/'payload'/'r03_css_addendum.css').read_text(encoding='utf-8')
    css_original=css.read_text(encoding='utf-8') if css.exists() else ''
    css_new=css_original.rstrip()+'\n\n'+extra
    try:
        helper_dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/'payload'/'r03_helpers.py',helper_dest)
        css.write_text(css_new,encoding='utf-8')
        # Temp + atomic replace prevents partial app.py writes.
        app_tmp=app.with_name('app.r03_tmp.py')
        app_tmp.write_text(modified,encoding='utf-8')
        ast.parse(app_tmp.read_text(encoding='utf-8'))
        app_tmp.replace(app)
    except Exception:
        shutil.copy2(backup/'app.py',app)
        if (backup/'uas_theme.css').exists():
            shutil.copy2(backup/'uas_theme.css',css)
        elif css.exists():
            css.unlink()
        if (backup/'r03_helpers.py').exists():
            shutil.copy2(backup/'r03_helpers.py',helper_dest)
        elif helper_dest.exists():
            helper_dest.unlink()
        if app.with_name('app.r03_tmp.py').exists():
            app.with_name('app.r03_tmp.py').unlink()
        raise
    manifest=dict(summary,backup=str(backup),before_app_sha256=sha256(backup/'app.py'),
                  after_app_sha256=sha256(app),checks=['python -m py_compile app.py','python tools/r03_verify_local.py --project .'])
    (backup/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print('R03_PATCH_APPLIED | Backup:',backup)
    print('Model, tokenizer, threshold, dataset, dan evaluator tidak ditulis/diubah.')


if __name__=='__main__':
    sys.exit(main())
