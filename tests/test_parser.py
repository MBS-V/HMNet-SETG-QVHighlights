from src.setg.parser import SubEventParser

def test_simple_fallback():
 p=SubEventParser(use_spacy=False).parse('A person opens the door.')
 assert p.num_subevents==1 and p.subevents[0]
def test_leading_after_split():
 p=SubEventParser(use_spacy=False).parse('After getting out the juicer, he juices the first orange half.')
 assert p.num_subevents==2, p.subevents

def test_then_split():
 p=SubEventParser(use_spacy=False).parse('He opens the fridge and then takes a bottle.')
 assert p.num_subevents==2
