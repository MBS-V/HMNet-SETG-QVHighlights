"""Analysis-only simple/compound duration matching helpers."""
from ..evaluation.duration_matching import duration_match_indices

def match_query_buckets(simple_rows,compound_rows,bins=8,seed=42):
    def total_duration(row):
        windows=row.get('timestamps',row.get('relevant_windows'))
        if len(windows)==2 and all(isinstance(x,(int,float)) for x in windows): windows=[windows]
        merged=[]
        for lo,hi in sorted((float(w[0]),float(w[1])) for w in windows):
            if merged and lo<=merged[-1][1]: merged[-1]=(merged[-1][0],max(merged[-1][1],hi))
            else: merged.append((lo,hi))
        return sum(hi-lo for lo,hi in merged)
    simple_d=[total_duration(r) for r in simple_rows]
    compound_d=[total_duration(r) for r in compound_rows]
    si,ci,report=duration_match_indices(simple_d,compound_d,bins,seed)
    return [simple_rows[i] for i in si],[compound_rows[i] for i in ci],report
