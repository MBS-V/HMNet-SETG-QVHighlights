"""Inter-annotator and parser agreement without a scikit-learn dependency."""
def cohen_kappa(a,b):
    if len(a)!=len(b) or not a: raise ValueError('Need equal non-empty annotation lists')
    labels=set(a)|set(b); n=len(a); po=sum(x==y for x,y in zip(a,b))/n
    pe=sum((sum(x==lab for x in a)/n)*(sum(y==lab for y in b)/n) for lab in labels)
    return 1.0 if pe==1.0 and po==1.0 else (po-pe)/(1-pe) if pe<1 else 0.0
def agreement(annotator_a,annotator_b,parser_labels=None):
    if len(annotator_a)!=len(annotator_b): raise ValueError('Annotator arrays must have equal length')
    consensus=[1 if (a==1 and b==1) else 0 if (a==0 and b==0) else None for a,b in zip(annotator_a,annotator_b)]
    valid=[i for i,x in enumerate(consensus) if x is not None]
    out={'cohen_kappa':cohen_kappa(annotator_a,annotator_b),'n':len(annotator_a),'consensus_n':len(valid)}
    if parser_labels is not None: out['parser_accuracy_on_consensus']=sum(consensus[i]==parser_labels[i] for i in valid)/len(valid) if valid else None
    return out
