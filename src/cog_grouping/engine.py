"""Explicit precision, correctly normalized gradient accumulation and evaluation."""

from contextlib import nullcontext
import math
import time

import torch
from torch.nn import functional as F


def precision_context(device, precision):
    """Reject unsupported precision requests instead of silently changing precision."""
    if precision == "fp32":
        return nullcontext()
    if device.type != "cuda":
        raise ValueError("bf16/fp16 training requires CUDA; use fp32 for CPU verification")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("this CUDA device does not support bf16")
    if precision not in ("bf16", "fp16"):
        raise ValueError("precision must be fp32, bf16 or fp16")
    return torch.autocast("cuda", dtype=torch.bfloat16 if precision == "bf16" else torch.float16)


@torch.no_grad()
def evaluate(model, loader, device, precision="fp32", return_predictions=False):
    """Compute sample-weighted accuracy and NLL without selecting a checkpoint."""
    model.eval()
    total_loss, correct, samples = 0.0, 0, 0
    predictions = []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with precision_context(device, precision):
            logits = model(images)
        logits = logits.float()
        if not torch.isfinite(logits).all():
            raise FloatingPointError("nonfinite evaluation logits")
        total_loss += float(F.cross_entropy(logits, labels, reduction="sum"))
        predicted = logits.argmax(1)
        correct += int((predicted == labels).sum())
        samples += labels.numel()
        if return_predictions:
            for truth, guess, probabilities in zip(labels.cpu(), predicted.cpu(), logits.softmax(1).cpu()):
                predictions.append({"label": int(truth), "prediction": int(guess),
                                    "probabilities": probabilities.tolist()})
    if samples == 0:
        raise ValueError("evaluation loader is empty")
    result = {"accuracy": correct / samples, "nll": total_loss / samples, "samples": samples}
    if return_predictions:
        result["predictions"] = predictions
    return result


def train_epoch(model, loader, optimizer, device, precision="fp32", accumulation_steps=1,
                scaler=None, max_grad_norm=1.0):
    """Match the sample-mean gradient even for short batches and tail accumulation."""
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be positive")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss, correct, samples, group_samples = 0.0, 0, 0, 0
    steps, skipped = 0, 0
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    nominal_samples = accumulation_steps * loader.batch_size
    for batch_index, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        with precision_context(device, precision):
            logits = model(images)
            loss_sum = F.cross_entropy(logits.float(), labels, reduction="sum")
            objective = loss_sum / nominal_samples
        if not torch.isfinite(loss_sum):
            raise FloatingPointError("nonfinite training loss")
        if scaler is not None and scaler.is_enabled():
            scaler.scale(objective).backward()
        else:
            objective.backward()
        count = labels.numel()
        group_samples += count
        samples += count
        total_loss += float(loss_sum.detach())
        correct += int((logits.detach().argmax(1) == labels).sum())
        boundary = (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == len(loader)
        if boundary:
            if scaler is not None and scaler.is_enabled():
                scaler.unscale_(optimizer)
            for parameter in trainable:
                if parameter.grad is not None:
                    parameter.grad.mul_(nominal_samples / group_samples)
            finite = all(torch.isfinite(p.grad).all().item() for p in trainable if p.grad is not None)
            if not finite and not (scaler is not None and scaler.is_enabled()):
                raise FloatingPointError("nonfinite unscaled gradient")
            if finite and max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
            if scaler is not None and scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
                skipped += int(not finite)
            else:
                optimizer.step()
            steps += int(finite)
            optimizer.zero_grad(set_to_none=True)
            group_samples = 0
    if samples == 0:
        raise ValueError("training loader is empty")
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    return {"accuracy": correct / samples, "nll": total_loss / samples, "samples": samples,
            "optimizer_steps": steps, "skipped_updates": skipped,
            "seconds": time.perf_counter() - start, "peak_cuda_allocated_bytes": peak}


def epoch_learning_rate(base_lr, epoch, epochs, warmup_epochs):
    """Define a simple epoch-based warmup/cosine schedule with explicit units."""
    if warmup_epochs and epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
    return base_lr * 0.5 * (1 + math.cos(math.pi * min(1, progress)))
