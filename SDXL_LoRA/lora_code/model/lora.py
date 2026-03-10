import torch
import torch.nn as nn


class LoRALinearLayer(nn.Module):
    """Линейный слой LoRA.

    Реализует LoRA в виде двух последовательных линейных слоёв: понижающего (down) и повышающего (up).
    Вместо обучения полной матрицы весов размером (in_features x out_features),
    обучаются две матрицы: down (in_features x rank) и up (rank x out_features),
    что значительно сокращает количество обучаемых параметров.

    Матрица down инициализируется нормальным распределением с std=1/rank,
    а матрица up — нулями, поэтому в начале обучения добавка LoRA равна нулю.

    Args:
        in_features (int): Размерность входного пространства.
        out_features (int): Размерность выходного пространства.
        rank (int): Ранг низкоранговой аппроксимации. Должен быть не больше
            min(in_features, out_features). По умолчанию 4.
    """

    def __init__(self, in_features, out_features, rank=4):
        super().__init__()

        if rank > min(in_features, out_features):
            raise ValueError(
                f"LoRA rank {rank} must be less or equal than {min(in_features, out_features)}"
            )
        self.rank = rank

        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states):
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        hidden_states = hidden_states.to(dtype)
        output = self.up(self.down(hidden_states))

        return output.to(orig_dtype)


class LoRACrossAttnProcessor(nn.Module):
    """Процессор cross-attention с LoRA-адаптацией.

    Заменяет стандартный процессор внимания, добавляя к каждой из линейных
    проекций (query, key, value, out) параллельную низкоранговую ветку LoRA.
    При прямом проходе результат основной проекции складывается с выходом
    соответствующего LoRA-слоя, умноженным на коэффициент масштабирования scale.

    Поддерживает как self-attention (когда encoder_hidden_states=None и в качестве
    ключей/значений используются сами hidden_states), так и cross-attention
    (когда ключи/значения формируются из encoder_hidden_states).

    Args:
        hidden_size (int): Размерность скрытых состояний модели.
        lora_linear_layer (type): Класс LoRA-слоя, используемый для создания
            низкоранговых адаптаций. По умолчанию LoRALinearLayer.
        cross_attention_dim (int, optional): Размерность входа энкодера для
            cross-attention. Если None, используется hidden_size (self-attention).
        rank (int): Ранг низкоранговой аппроксимации. По умолчанию 4.
    """

    def __init__(
        self,
        hidden_size,
        lora_linear_layer=LoRALinearLayer,
        cross_attention_dim=None,
        rank=4,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        context_dim = cross_attention_dim or hidden_size

        self.to_q_lora = lora_linear_layer(hidden_size, hidden_size, rank=rank)
        self.to_k_lora = lora_linear_layer(context_dim, hidden_size, rank=rank)
        self.to_v_lora = lora_linear_layer(context_dim, hidden_size, rank=rank)
        self.to_out_lora = lora_linear_layer(hidden_size, hidden_size, rank=rank)

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        scale=1.0,
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )

        encoder_hidden_states = (
            encoder_hidden_states
            if encoder_hidden_states is not None
            else hidden_states
        )

        query = attn.to_q(hidden_states) + scale * self.to_q_lora(hidden_states)
        key = attn.to_k(encoder_hidden_states) + scale * self.to_k_lora(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states) + scale * self.to_v_lora(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        hidden_states = attn.to_out[0](hidden_states) + scale * self.to_out_lora(hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
