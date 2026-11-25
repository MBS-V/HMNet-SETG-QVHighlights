from src.setg.token_alignment import regex_token_offsets,char_spans_to_token_spans,align_query_tokens

def test_offsets_repeated_words():
 q='He sees a cat, then sees the cat.'; off=regex_token_offsets(q)
 spans=char_spans_to_token_spans([(0,13),(20,len(q))],off)
 assert spans[0][0]==0 and spans[1][1]==len(off)
def test_alignment():
 assert align_query_tokens('red red car',['red','red','car'])==[(0,3),(4,7),(8,11)]
