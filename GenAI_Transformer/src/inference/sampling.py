"""
Sampling strategies for music generation.

- temperature_sampling: Điều chỉnh diversity qua temperature
- top_k_sampling: Chỉ sample từ top-k tokens
- top_p_sampling: Nucleus sampling — tự động chọn tập tokens
- combined_sampling: Top-p + Temperature (đề xuất cho music)
"""

import torch
import torch.nn.functional as F


def temperature_sampling(
    logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Temperature Sampling.

    Điều chỉnh "độ sắc" của phân phối xác suất:
    - τ → 0: Greedy (luôn chọn token xác suất cao nhất)
    - τ = 1: Phân phối gốc
    - τ > 1: Phân phối phẳng hơn (diverse hơn)

    Args:
        logits: (vocab_size,) — raw logits từ model
        temperature: float — diversity control

    Returns:
        (1,) — sampled token ID
    """
    if temperature <= 0:
        # Greedy
        return logits.argmax(dim=-1, keepdim=True)

    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def top_k_sampling(
    logits: torch.Tensor,
    k: int = 50,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Top-k Sampling (Fan et al., 2018).

    Chỉ giữ lại k tokens có xác suất cao nhất,
    đặt xác suất còn lại = 0, rồi sample.

    Args:
        logits: (vocab_size,) — raw logits
        k: int — số tokens giữ lại
        temperature: float — temperature scaling

    Returns:
        (1,) — sampled token ID
    """
    if temperature > 0:
        logits = logits / temperature

    # Tìm top-k values
    top_k_values, top_k_indices = torch.topk(logits, min(k, logits.size(-1)))

    # Mask tất cả giá trị khác = -inf
    logits_filtered = torch.full_like(logits, float("-inf"))
    logits_filtered.scatter_(0, top_k_indices, top_k_values)

    probs = F.softmax(logits_filtered, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def top_p_sampling(
    logits: torch.Tensor,
    p: float = 0.9,
    temperature: float = 1.0,
    min_tokens_to_keep: int = 1,
) -> torch.Tensor:
    """
    Top-p (Nucleus) Sampling (Holtzman et al., 2020).

    Chọn tập tokens nhỏ nhất sao cho tổng xác suất ≥ p.
    Tự động điều chỉnh số tokens dựa trên phân phối:
    - Khi model tự tin → ít tokens
    - Khi model không chắc chắn → nhiều tokens

    Đây là phương pháp đề xuất cho music generation.

    Args:
        logits: (vocab_size,) — raw logits
        p: float — nucleus probability threshold
        temperature: float — temperature scaling
        min_tokens_to_keep: int — tối thiểu tokens giữ lại

    Returns:
        (1,) — sampled token ID
    """
    if temperature > 0:
        logits = logits / temperature

    # Sort theo xác suất giảm dần
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=-1)

    # Cumulative probability
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Mask: loại bỏ tokens có cumulative prob > p
    # Shift right 1 để giữ token đầu tiên vượt p
    sorted_mask = cumulative_probs - sorted_probs > p

    # Đảm bảo giữ ít nhất min_tokens_to_keep
    if min_tokens_to_keep > 0:
        sorted_mask[:min_tokens_to_keep] = False

    # Mask logits
    sorted_logits[sorted_mask] = float("-inf")

    # Re-sort về thứ tự gốc
    logits_filtered = torch.zeros_like(logits)
    logits_filtered.scatter_(0, sorted_indices, sorted_logits)

    probs = F.softmax(logits_filtered, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def combined_sampling(
    logits: torch.Tensor,
    temperature: float = 0.85,
    top_p: float = 0.9,
    top_k: int = 0,
    min_tokens_to_keep: int = 1,
) -> torch.Tensor:
    """
    Combined sampling: Temperature + Top-p + (optional) Top-k.

    Đề xuất cho music generation:
    - temperature=0.85: đủ diverse nhưng vẫn coherent
    - top_p=0.9: loại bỏ tokens xác suất rất thấp

    Args:
        logits: (vocab_size,) — raw logits
        temperature: float — diversity control
        top_p: float — nucleus threshold (0 = disabled)
        top_k: int — top-k filtering (0 = disabled)
        min_tokens_to_keep: int — tối thiểu tokens giữ lại

    Returns:
        (1,) — sampled token ID
    """
    # Temperature scaling
    if temperature > 0:
        logits = logits / temperature
    else:
        return logits.argmax(dim=-1, keepdim=True)

    # Top-k filtering (optional)
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][-1]
        logits[indices_to_remove] = float("-inf")

    # Top-p (nucleus) filtering
    if top_p > 0 and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_mask = cumulative_probs - sorted_probs > top_p
        if min_tokens_to_keep > 0:
            sorted_mask[:min_tokens_to_keep] = False

        # Map mask back to original indices
        indices_to_remove = sorted_mask.scatter(0, sorted_indices, sorted_mask)
        logits[indices_to_remove] = float("-inf")

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
