def compute_confidence_score(logits, input_ids):
    import torch.nn.functional as F
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="none")
    token_loss = loss.view(shift_labels.size())
    avg_confidence = torch.exp(-token_loss).mean().item()
    return avg_confidence
