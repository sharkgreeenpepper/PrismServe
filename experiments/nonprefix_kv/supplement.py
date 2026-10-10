"""Select full-context LongBench-v2 multi-document items; never truncate evidence."""
import argparse
import hashlib
import json
from pathlib import Path
from transformers import AutoTokenizer
from build_data import token_hash


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--model',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();t=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    source=json.loads(a.source.read_text());rows=[]
    candidates=sorted([r for r in source if r['domain']=='Multi-Document QA'],
                      key=lambda r:(len(r['context']),r['_id']))
    for r in candidates:
        if len(rows)>=20:break
        if len(r['context'])>100000:break
        question=r['question']+'\n'+'\n'.join(f"{c}. {r['choice_'+c]}" for c in 'ABCD')
        messages=[{'role':'system','content':'Read the full evidence and choose the correct option. Return only A, B, C, or D.'},
                  {'role':'user','content':r['context']+'\n\nQUESTION:\n'+question}]
        rendered=t.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        encoded=t(rendered,add_special_tokens=False,return_offsets_mapping=True)
        ids=encoded['input_ids'];actual=len(ids)
        if actual>16384:continue
        start=rendered.rindex('QUESTION:')
        at=next(i for i,(_,end) in enumerate(encoded['offset_mapping']) if end>start)
        ids[at:at]=[t.encode(' archive',add_special_tokens=False)[0]]*(16384-actual)
        rows.append(dict(request_id=f'longbench-{r["_id"]}',session_id=f'longbench-{r["_id"]}',
                         kind='longbench_multi_document',step=0,length=16384,split='supplement',
                         answer=r['answer'],answer_type='choice',prompt_token_ids=ids,input_hash=token_hash(ids),
                         original_tokens=actual,original_id=r['_id'],difficulty=r['difficulty'],
                         context_sha256=hashlib.sha256(r['context'].encode()).hexdigest(),
                         source=str(a.source),context_truncated=False))
    a.output.write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows))
    a.output.with_suffix('.manifest.json').write_text(json.dumps(
        {'n':len(rows),'source':str(a.source),'selection':'shortest complete Multi-Document QA fitting 16K; max 20',
         'truncated':False,'padding':'inert archive tokens before question','not_representative':True},indent=2))
    print('SUPPLEMENT',len(rows))
