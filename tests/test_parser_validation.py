from src.evaluation.parser_validation import agreement

def test_cohen_kappa_and_parser_consensus():
 r=agreement([0,1,1,0],[0,1,0,0],[0,1,1,0])
 assert r['cohen_kappa']>0 and r['consensus_n']==3 and r['parser_accuracy_on_consensus']==1.0
