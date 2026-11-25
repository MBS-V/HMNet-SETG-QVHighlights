def difference_in_differences(hm_simple,setg_simple,hm_compound,setg_compound):
    ds=setg_simple-hm_simple; dc=setg_compound-hm_compound
    return {'simple_improvement':ds,'compound_improvement':dc,'difference_in_differences':dc-ds}
