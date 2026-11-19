"""Equation (1)-(5) multi-level probability-map distillation and CTC loss."""
import torch
import torch.nn.functional as F


def _align(prob, target_time):
    if prob.size(1) == target_time:
        return prob
    return F.interpolate(prob.transpose(1, 2), size=target_time, mode="linear", align_corners=False).transpose(1, 2)


def _kl(student_prob, teacher_prob, temperature):
    # Sum over gloss classes, average over batch/time, with the paper's alpha=T multiplier.
    # Clamp avoids log(0) for probability-map APIs; gradients still pass through student.
    student_prob = student_prob.clamp_min(1e-8)
    teacher_prob = teacher_prob.detach().clamp_min(1e-8)
    return temperature * (teacher_prob * (teacher_prob.log() - student_prob.log())).sum(-1).mean()


def kd_mslrt_loss(student, teacher, targets, input_lengths, target_lengths,
                  blank=0, temperature=1.0, kd_weight=25.0):
    """Return total and named losses. Teacher/student dicts expose both probability maps."""
    lc = _kl(student["conv_probs"], _align(teacher["conv_probs"], student["conv_probs"].size(1)), temperature)
    lb = _kl(student["lstm_probs"], _align(teacher["lstm_probs"], student["lstm_probs"].size(1)), temperature)
    ls = _kl(student["conv_probs"], student["lstm_probs"], temperature)
    lengths = student.get("output_lengths")
    if lengths is None:
        lengths = input_lengths
    log_probs = student["lstm_logits"].log_softmax(-1).transpose(0, 1)
    ctc = F.ctc_loss(log_probs, targets, lengths.cpu(), target_lengths.cpu(), blank=blank, zero_infinity=True)
    kd = lc + lb + ls
    return {"total": ctc + kd_weight * kd, "ctc": ctc, "kd": kd, "lc": lc, "lb": lb, "ls": ls}
