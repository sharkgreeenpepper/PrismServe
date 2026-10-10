"""Independent prompt-label and exact token/span checks on the frozen corpus."""
import argparse
import bisect
import hashlib
import itertools
import json
import re
from pathlib import Path
from transformers import AutoTokenizer
from build_data import token_hash


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--model',required=True);p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();t=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    rows=[json.loads(s) for s in a.data.read_text().splitlines()]
    token_pieces={};seen=set();errors=[];span_corrections=0
    for r in rows:
        ids=r['prompt_token_ids'];seen.add(r['request_id'])
        if len(ids)!=r['length'] or token_hash(ids)!=r['input_hash']:
            errors.append((r['request_id'],'token/hash mismatch'));continue
        text=t.decode(ids)
        header=f'Document {r["session_id"]}/record-{r["target_entity"]:02d}.'
        segment=text.split(header)[1].split('Document ')[0]
        observed=re.search(r'Current revision (\d+) amount (\d{4})\.',segment)
        question=re.search(r'What is the current amount for ITEM-(\d{2})\?',text)
        if (observed is None or observed.group(2)!=r['answer'] or
            question is None or int(question.group(1))!=r['target_entity']):
            errors.append((r['request_id'],'independent prompt oracle disagrees'))
        # ASCII corpus: singleton token decoding gives exact character boundaries.
        pieces=[]
        for tid in ids:
            if tid not in token_pieces:token_pieces[tid]=t.decode([tid])
            pieces.append(token_pieces[tid])
        assert ''.join(pieces)==text
        ends=list(itertools.accumulate(map(len,pieces)))
        starts=[m.start() for m in re.finditer(r'Document '+re.escape(r['session_id'])+r'/record-\d{2}\.',text)]
        assert len(starts)==16
        bounds=[0]+[bisect.bisect_right(ends,s) for s in starts]
        qstart=text.rindex('What is the current amount')
        bounds+=[bisect.bisect_right(ends,qstart),len(ids)]
        spans=list(zip(bounds[:-1],bounds[1:]))
        if r['document_spans']!=[list(x) for x in spans]:span_corrections+=1
        r['document_spans']=spans
    assert len(seen)==len(rows)
    if errors:raise AssertionError(errors[:10])
    # Metadata-only correction: token IDs and input hashes remain identical.
    temporary=a.data.with_suffix('.audited.tmp')
    temporary.write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows))
    temporary.replace(a.data)
    report={'n':len(rows),'errors':errors,'all_input_hashes_unchanged':True,
            'independent_label_check':'current amount parsed from actual model prompt',
            'span_metadata_corrected':span_corrections,
            'sha256':hashlib.sha256(a.data.read_bytes()).hexdigest()}
    a.report.write_text(json.dumps(report,indent=2));print(report)
