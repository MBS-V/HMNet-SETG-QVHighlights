"""Interpretable sub-event parser. spaCy is optional; deterministic fallback is explicit."""
from dataclasses import dataclass, asdict
import re

CONNECTIVES=("and then","followed by","after","before","then","once","while")
@dataclass
class ParseResult:
    query: str
    subevents: list
    char_spans: list
    num_subevents: int
    status: str
    warnings: list
    truncated: bool=False
    def to_dict(self): return asdict(self)

class SubEventParser:
    def __init__(self,max_subevents=4, use_spacy=True):
        self.max_subevents=max(1,int(max_subevents)); self.nlp=None; self.status="regex_fallback"
        if use_spacy:
            try:
                import spacy
                self.nlp=spacy.load("en_core_web_sm"); self.status="spacy"
            except Exception:
                self.nlp=None

    def parse(self, query: str) -> ParseResult:
        q=query.strip(); warnings=[]
        if not q: return ParseResult(query, [query], [(0,len(query))], 1, "empty_query_fallback", ["Empty query"])
        spans=[]
        if self.nlp is not None:
            try:
                doc=self.nlp(q)
                # First split on explicit sequencing connectives, preserving text and surface order.
                spans=self._split_connectives(q)
                if len(spans)==1:
                    verbs=sorted([t for t in doc if t.pos_=="VERB" and (t.dep_ in {"ROOT","conj"})],key=lambda t:t.i)
                    if len(verbs)>1:
                        # Split at each coordinated verb; include a coordinating word (and/but/or)
                        # with the following clause, preserving surface order and arguments.
                        cuts=[]
                        for prev,verb in zip(verbs[:-1],verbs[1:]):
                            cc=[t for t in doc[prev.i+1:verb.i] if t.dep_=="cc"]
                            cuts.append(cc[-1].idx if cc else verb.idx)
                        starts=[0]+cuts; ends=cuts+[len(q)]
                        spans=list(zip(starts,ends))
            except Exception as exc:
                warnings.append(f"spaCy parsing failed: {type(exc).__name__}: {exc}")
        else:
            spans=self._split_connectives(q)
        spans=[(a,b) for a,b in spans if q[a:b].strip()]
        if not spans: spans=[(0,len(q))]
        spans=self._merge_overlaps(spans)
        # A weak connective split that creates fragments without a verb is still retained;
        # the output is a transparent heuristic, not a semantic gold parser.
        truncated=len(spans)>self.max_subevents
        if truncated:
            spans=spans[:self.max_subevents]; warnings.append(f"Truncated to max_subevents={self.max_subevents}")
        texts=[q[a:b].strip(" ,;.") for a,b in spans]
        clean=[]
        for t,(a,b) in zip(texts,spans):
            if t and (not clean or t.lower()!=clean[-1].lower()): clean.append(t)
        if len(clean)!=len(texts):
            spans=[(0,len(q))]; clean=[q]
        status="spacy" if self.nlp is not None else "regex_fallback"
        return ParseResult(q,clean,spans,len(clean),status,warnings,truncated)

    @staticmethod
    def _split_connectives(q):
        pattern=re.compile(r"\s+(?:and\s+then|followed\s+by|after|before|then|once|while)\s+",re.I)
        # Leading subordinating connective: "After A, B" is two clauses even though
        # the connective is at character zero and would not match the generic pattern.
        leading=re.match(r"^(?:after|before|once|while)\s+(.+?),\s+(.+)$",q,re.I)
        if leading:
            comma=q.find(',',leading.start(1))
            return [(0,comma),(comma+1,len(q))]
        matches=list(pattern.finditer(q))
        if not matches: return [(0,len(q))]
        spans=[]; start=0
        for m in matches:
            if m.start()>start: spans.append((start,m.start()))
            # Keep connective attached to the following clause for grammatical context.
            start=m.start()
        spans.append((start,len(q)))
        return spans if len(spans)>1 else [(0,len(q))]

    @staticmethod
    def _merge_overlaps(spans):
        out=[]
        for a,b in sorted(spans):
            if out and a<out[-1][1]: out[-1]=(out[-1][0],max(out[-1][1],b))
            else: out.append((a,b))
        return out
