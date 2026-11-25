"""Map parser character spans to model-token indices using tokenizer offsets."""
import re

def regex_token_offsets(text):
    return [(m.start(),m.end()) for m in re.finditer(r"\w+|[^\w\s]",text)]

def char_spans_to_token_spans(char_spans, token_offsets):
    result=[]
    for a,b in char_spans:
        ids=[i for i,(s,e) in enumerate(token_offsets) if e>a and s<b]
        if ids: result.append((min(ids),max(ids)+1))
        else: result.append((0,0))
    return result

def align_query_tokens(query, model_tokens):
    """Best-effort exact surface alignment; prefer offsets returned by the feature tokenizer.

    Returns character offsets. If token strings do not reconstruct the query, raises
    ValueError rather than silently aligning the wrong repeated word.
    """
    offsets=[]; cursor=0
    for tok in model_tokens:
        t=str(tok).replace("##","").replace("▁","").strip()
        if not t: offsets.append((cursor,cursor)); continue
        idx=query.lower().find(t.lower(),cursor)
        if idx<0: raise ValueError(f"Cannot align token {tok!r} after character {cursor}; supply tokenizer offset_mapping")
        offsets.append((idx,idx+len(t))); cursor=idx+len(t)
    return offsets
