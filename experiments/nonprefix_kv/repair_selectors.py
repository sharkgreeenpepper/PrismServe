"""Equal-budget selection logic, staged for an engine adapter; NOT a KV runtime.

This module never pretends to load/recompute KV. P1/P2 GPU integration remains
blocked until the public layerwise backend passes its compatibility witness.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RepairSelection:
    repair_positions: tuple[int, ...]
    mandatory_positions: tuple[int, ...]
    reusable_tokens: int
    budget: int
    fallback_full: bool = False
    reason: str | None = None


def best_window(positions, scores, count):
    """Window covering exactly count reusable tokens; holes are mandatory compute."""
    if count == 0:
        return []
    if len(positions) < count:
        raise ValueError('Insufficient reusable tokens')
    prefix=[0.0]
    for pos in positions:prefix.append(prefix[-1]+scores[pos])
    # Tie-break deterministically: shortest physical window, then earliest start.
    start=max(range(len(positions)-count+1),key=lambda i:(
        prefix[i+count]-prefix[i],-(positions[i+count-1]-positions[i]),-positions[i]))
    return positions[start:start+count]


def select_repair(scores, reusable, ratio, policy, document_spans=(),
                  edit_start=None, dependency_end=None):
    if len(scores)!=len(reusable) or not 0<=ratio<=1:
        raise ValueError('Invalid score/mask/ratio')
    n=len(scores)
    positions=[i for i,available in enumerate(reusable) if available]
    mandatory=tuple(i for i,available in enumerate(reusable) if not available)
    budget=min(len(positions),max(1,math.floor(ratio*len(positions)))) if ratio>0 and positions else 0
    def full(reason):
        return RepairSelection(tuple(positions),mandatory,len(positions),budget,True,reason)
    if any(not math.isfinite(s) or s<0 for s in scores):
        return full('Invalid deviation scores')
    if policy not in ('topk','window','document','edit_window'):
        raise ValueError('Unknown selector')
    if not positions or budget==0:
        return RepairSelection((),mandatory,len(positions),budget)
    if policy=='topk':
        chosen=sorted(positions,key=lambda i:(-scores[i],i))[:budget]
    elif policy=='window':
        chosen=best_window(positions,scores,budget)
    elif policy=='document':
        spans=sorted(document_spans)
        if any(not 0<=a<b<=n for a,b in spans) or any(spans[i][1]>spans[i+1][0] for i in range(len(spans)-1)):
            raise ValueError('Invalid or overlapping document spans')
        candidates=[]
        covered=set()
        for a,b in spans:
            members=[i for i in positions if a<=i<b]
            if members:
                covered.update(members)
                candidates.append((sum(scores[i] for i in members)/len(members),a,members))
        if covered!=set(positions):
            return full('Document metadata does not cover reusable positions')
        candidates.sort(key=lambda item:(-item[0],item[1]))
        chosen=[];remaining=budget;deferred=[]
        for _,_,members in candidates:
            if len(members)<=remaining:
                chosen.extend(members);remaining-=len(members)
            else:deferred.append(members)
        if remaining:
            chosen.extend(best_window(deferred[0],scores,remaining))
    else:
        if edit_start is None or dependency_end is None:
            return full('Causal dependency extent is unknown')
        if not 0<=edit_start<dependency_end<=n:
            raise ValueError('Invalid causal dependency interval')
        downstream=[i for i in positions if i>=edit_start]
        required=sum(edit_start<=i<dependency_end for i in positions)
        if required>budget:
            return full('Causal dependency interval exceeds repair budget')
        if len(downstream)<budget:
            return full('Insufficient downstream positions for equal-budget repair')
        chosen=downstream[:budget]
    chosen=tuple(sorted(chosen))
    assert len(chosen)==budget and len(set(chosen))==budget
    assert set(chosen).isdisjoint(mandatory)
    return RepairSelection(chosen,mandatory,len(positions),budget)
