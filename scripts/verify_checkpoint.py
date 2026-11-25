import argparse,torch
p=argparse.ArgumentParser(); p.add_argument('checkpoint'); a=p.parse_args(); x=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
print('checkpoint readable:',True,'keys:',sorted(x.keys())); print('epoch:',x.get('epoch'),'global_step:',x.get('global_step'))
