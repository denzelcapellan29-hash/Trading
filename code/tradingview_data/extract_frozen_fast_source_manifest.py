#!/usr/bin/env python3
"""Derive source symbols from exact frozen 31 FAST Pine scripts; no manual substitutes."""
import argparse, json, re, zipfile
from pathlib import Path

def extract(zip_file):
    with zipfile.ZipFile(zip_file) as archive:
        files=[p for p in archive.namelist() if re.search(r'/pine/[A-Z]{6}\.pine$',p)]
        model=set(); factors=set()
        for path in files:
            pair=Path(path).stem
            src=archive.read(path).decode()
            for alias,symbol in re.findall(r'string\s+sym([A-Za-z0-9]+)\s*=\s*input\.symbol\("([^"]+)',src):
                (model if symbol.startswith('FX:') else factors).add(symbol)
            assert f'FX:{pair}' in model, f'Missing model spot for {pair}'
    if len(files)!=31 or len(model)!=31 or len(factors)!=66:
        raise AssertionError(f'Unexpected canonical count: {len(files)}, {len(model)}, {len(factors)}')
    return {'freeze_id':'FX-FAST-2026-08-28',
            'provenance':'Frozen 2026-08-28 handoff exact Pine input.symbol defaults',
            'fxcm_model_spot': sorted(model),'factor_symbols':sorted(factors)}

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('handoff',type=Path);a.add_argument('--out',type=Path,required=True)
    args=a.parse_args();out=extract(args.handoff);args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(out,indent=2)+'\n')
    print('FXCM model spot:',len(out['fxcm_model_spot']),'factors:',len(out['factor_symbols']))
