import re

import pytest
import torch
import torch.nn.functional as F

from src.caduceus_distillation import distillation_loss


@pytest.fixture
def basic_inputs():
    student = torch.randn(2, 5, 10)
    teacher = torch.randn(2, 5, 10)
    targets = torch.randint(0, 10, (2, 5))
    return student, teacher, targets


def test_happy_path(basic_inputs):
    student, teacher, targets = basic_inputs
    loss = distillation_loss(student, teacher, targets)
    assert torch.isfinite(loss)
    assert loss.shape == ()


def test_temperature_scaling(basic_inputs):
    student, teacher, targets = basic_inputs
    loss1 = distillation_loss(student, teacher, targets, temperature=1.0)
    loss4 = distillation_loss(student, teacher, targets, temperature=4.0)
    assert not torch.allclose(loss1, loss4)

    # NOTE: If temp tends to infinity, softmax should be close to uniform,
    # and with alpha=1.0 (soft target loss only), the loss should be 0
    loss_high_temp = distillation_loss(
        student, teacher, targets, temperature=1e9, alpha=1.0
    )
    assert torch.isclose(loss_high_temp, torch.tensor(0.0))

    # NOTE: if temp is very low, softmax should be close to one-hot encoding,
    # and so soft loss will tend to hard loss for scaled logits
    small_temp = 1e-9
    loss_low_temp = distillation_loss(
        student, teacher, targets, temperature=small_temp, alpha=1.0
    )
    hard_loss = F.cross_entropy(
        (student / small_temp).view(-1, student.size(-1)),
        torch.argmax(teacher, dim=-1).view(-1),
        # NOTE: match the `batchmean`` behavior of KL divergence, first sum then divide by batch size
        reduction="sum",
    ) / (student.size(0) * student.size(1))
    assert torch.isclose(loss_low_temp, hard_loss)


def test_invalid_inputs(basic_inputs):
    student, teacher, targets = basic_inputs
    with pytest.raises(
        AssertionError, match="Expected student_logits to be 3D, got 2D"
    ):
        distillation_loss(student[0], teacher, targets)
    with pytest.raises(AssertionError, match="Temperature must be positive"):
        distillation_loss(student, teacher, targets, temperature=-1.0)
    with pytest.raises(AssertionError, match=re.escape("Alpha must be in [0, 1]")):
        distillation_loss(student, teacher, targets, alpha=1.1)
