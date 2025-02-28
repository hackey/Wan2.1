# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import warnings

__all__ = [
    'attention',
]


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),  # Не используется, оставлено для совместимости
    deterministic=False,   # Не используется, оставлено для совместимости
    dtype=torch.bfloat16,
    training=False,        # Новый параметр для режима обучения
    fa_version=None,       # Не используется, оставлено для совместимости
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). Ignored in this implementation.
    deterministic:  bool. Ignored in this implementation.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    training:       bool. Whether the model is in training mode.
    """
    # Предупреждение о неподдерживаемых функциях
    if window_size != (-1, -1):
        warnings.warn(
            "Windowed attention is not supported in this implementation. Using global attention instead."
        )
    if deterministic:
        warnings.warn(
            "Deterministic mode is not supported in this implementation. Ignoring."
        )

    # Проверяем, что все тензоры находятся на одном устройстве
    device = q.device
    if k.device != device or v.device != device:
        raise RuntimeError(
            f"All input tensors (q, k, v) must be on the same device, but found {q.device}, {k.device}, {v.device}"
        )

    # Применяем масштаб query, если указан
    if q_scale is not None:
        q = q * q_scale

    # Приводим тип данных
    q = q.to(dtype)
    k = k.to(dtype)
    v = v.to(dtype)

    # Создаём маску для переменной длины последовательностей
    if k_lens is not None:
        k_lens = k_lens.to(device)  # Переносим k_lens на устройство k
        max_k_len = k.size(1)
        mask = torch.arange(max_k_len, device=device)[None, :] < k_lens[:, None]
        mask = mask[:, None, :].expand(-1, q.size(1), -1)  # [B, Lq, Lk]
        attn_mask = torch.where(mask, 0.0, float('-inf'))
    else:
        attn_mask = None

    # Транспонируем для scaled_dot_product_attention
    q = q.transpose(1, 2)  # [B, Nq, Lq, C1]
    k = k.transpose(1, 2)  # [B, Nk, Lk, C1]
    v = v.transpose(1, 2)  # [B, Nk, Lk, C2]

    # Выполняем attention
    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attn_mask,
        dropout_p=dropout_p if training else 0.0,  # Используем training
        is_causal=causal,
        scale=softmax_scale
    )

    # Возвращаем в исходный формат
    out = out.transpose(1, 2).contiguous()  # [B, Lq, Nq, C2]
    return out
