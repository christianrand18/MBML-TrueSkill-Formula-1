import torch


def plackett_luce_log_prob(
    performances: torch.Tensor,
    race_lengths: torch.Tensor,
) -> torch.Tensor:
    R = len(race_lengths)
    max_N = race_lengths.max().item()
    device = performances.device

    padded = torch.full((R, max_N), float("-inf"), device=device)
    offsets = torch.zeros(R, dtype=torch.long, device=device)
    offsets[1:] = torch.cumsum(race_lengths, dim=0)[:-1]
    for r in range(R):
        L = race_lengths[r].item()
        padded[r, :L] = performances[offsets[r] : offsets[r] + L]

    i_range = torch.arange(max_N, device=device)
    valid = i_range.unsqueeze(0) < race_lengths.unsqueeze(1)

    log_p_matrix = torch.zeros(R, max_N, device=device)
    for i in range(max_N):
        tail_lse = torch.logsumexp(padded[:, i:], dim=1)
        log_p_i = padded[:, i] - tail_lse
        log_p_matrix[:, i] = torch.where(valid[:, i], log_p_i, torch.zeros_like(log_p_i))

    return log_p_matrix[valid].sum()
