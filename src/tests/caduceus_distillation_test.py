import re

import pytest
import torch
import torch.nn.functional as F

from src.caduceus_distillation import (
    CADUCEUS_PAD_TOKEN_ID,
    _filter_non_specific_nucleotides_and_batch,
    distillation_loss,
)


@pytest.fixture
def basic_inputs():
    B, S, V = 2, 5, 16  # Batch size, Sequence length, Vocabulary size
    student = torch.randn(B, S, V)
    teacher = torch.randn(B, S, V)
    # 7 to 10 represent the valid nucleotide token IDs, it doesn't really matter for the test
    # but we do test some `non-specific` nucleotide (token ID 11)
    targets = torch.randint(7, 11, (B, S))
    return student, teacher, targets


def test_happy_path(basic_inputs):
    student, teacher, targets = basic_inputs
    loss, _, _ = distillation_loss(student, teacher, targets)
    assert torch.isfinite(loss)
    assert loss.shape == ()


def test_temperature_scaling(basic_inputs):
    student, teacher, targets = basic_inputs
    loss1, _, _ = distillation_loss(student, teacher, targets, temperature=1.0)
    loss4, _, _ = distillation_loss(student, teacher, targets, temperature=4.0)
    assert not torch.allclose(loss1, loss4)

    # NOTE: If temp tends to infinity, softmax should be close to uniform,
    # and with alpha=1.0 (soft target loss only), the loss should be 0
    loss_high_temp, _, _ = distillation_loss(
        student, teacher, targets, temperature=1e9, alpha=1.0
    )
    assert torch.isclose(loss_high_temp, torch.tensor(0.0))

    # NOTE: if temp is very low, softmax should be close to one-hot encoding,
    # and so soft loss will tend to hard loss for scaled logits
    small_temp = 1e-9
    loss_low_temp, _, _ = distillation_loss(
        student, teacher, targets, temperature=small_temp, alpha=1.0
    )
    hard_loss = F.cross_entropy(
        (student / small_temp).view(-1, student.size(-1)),
        torch.argmax(teacher, dim=-1).view(-1),
    )
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


def test_filter_non_specific_nucleotides(basic_inputs):
    student, teacher, targets = basic_inputs

    original_num_targests = targets.numel()
    # randomly set 10% of the targets to a special token CADUCEUS_NON_SPECIFIC_NUCLEOTIDE_TOKEN_ID
    num_non_specific = int(0.1 * targets.numel())
    non_specific_indices = torch.randperm(targets.numel())[:num_non_specific]
    targets.view(-1)[non_specific_indices] = CADUCEUS_PAD_TOKEN_ID

    assert torch.any(targets == CADUCEUS_PAD_TOKEN_ID)

    assert student.ndim == 3
    assert teacher.ndim == 3
    assert targets.ndim == 2

    student, teacher, targets = _filter_non_specific_nucleotides_and_batch(
        student, teacher, targets
    )

    # Check that the non-specific nucleotide token is filtered out
    assert not torch.any(targets == CADUCEUS_PAD_TOKEN_ID)
    assert student.ndim == 2
    assert teacher.ndim == 2
    assert targets.ndim == 1
    assert student.size(0) == teacher.size(0) == targets.size(0)
    assert student.size(1) == teacher.size(1)

    assert original_num_targests - num_non_specific == student.size(0)


def test_filter_non_specific_nucleotides_filter_all(basic_inputs):
    student, teacher, targets = basic_inputs

    # Set all targets to a non-specific nucleotide token
    targets.fill_(CADUCEUS_PAD_TOKEN_ID)

    student, teacher, targets = _filter_non_specific_nucleotides_and_batch(
        student, teacher, targets
    )

    assert targets.size(0) == 0
    assert student.size(0) == 0
