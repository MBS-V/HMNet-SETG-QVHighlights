from .localization import localization_loss
from .subevent_losses import containment_loss, diversity_loss
from .saliency import saliency_max_margin_loss

def compute_total_loss(outputs,targets,gt_spans=None,video_mask=None,weight_sub=0.1,weight_div=0.1,
                       weight_ce=6.,weight_l1=10.,weight_giou=1.,weight_sal=1.0,weight_rec=0.1,saliency_margin=0.2,saliency_scores=None,video_ids=None,saliency_loss_fn=None,reconstruction_loss_fn=None,eos_coef=1.0):
    loc,parts=localization_loss(outputs,targets,weight_ce,weight_l1,weight_giou,eos_coef)
    total=loc; parts['loss_localization']=loc.detach()
    if 'subevent_alignments' in outputs and gt_spans is not None:
        lsub=containment_loss(outputs['subevent_alignments'],outputs['subevent_mask'],gt_spans,video_mask)
        ldiv=diversity_loss(outputs['subevent_alignments'],outputs['subevent_mask'])
    else:
        lsub=loc*0; ldiv=loc*0
    total=total+weight_sub*lsub+weight_div*ldiv
    parts.update(loss_sub=lsub.detach(),loss_div=ldiv.detach())
    if saliency_scores is not None and 'video_memory' in outputs and 'query_embedding' in outputs:
        ls=saliency_max_margin_loss(outputs['video_memory'],outputs['query_embedding'],saliency_scores,video_mask,
                                    saliency_margin,video_ids)
        total=total+weight_sal*ls; parts['loss_saliency']=ls.detach()
    if 'loss_reconstruction' in outputs:
        lr=outputs['loss_reconstruction']; total=total+weight_rec*lr
        parts['loss_reconstruction']=lr.detach()
        if 'loss_fw' in outputs: parts['loss_fw']=outputs['loss_fw'].detach()
        if 'loss_ss' in outputs: parts['loss_ss']=outputs['loss_ss'].detach()
    # Optional losses must be supplied by a benchmark-faithful adapter; never fabricate labels.
    if saliency_loss_fn is not None:
        ls=saliency_loss_fn(outputs,targets); total=total+ls; parts['loss_saliency']=ls.detach()
    if reconstruction_loss_fn is not None:
        lr=reconstruction_loss_fn(outputs,targets); total=total+0.1*lr; parts['loss_reconstruction']=lr.detach()
    return total,parts
