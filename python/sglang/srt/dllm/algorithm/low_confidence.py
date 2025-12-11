from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner


class LowConfidence(DllmAlgorithm):
    """LowConfidence algorithm with step_map support."""

    def __init__(self, config: DllmConfig):
        super().__init__(config)
        self.threshold = self.algorithm_config.get("threshold", 0.95)

        # Request state management: req_pool_idx -> {step, initial_mask_count, ...}
        # State is automatically cleaned up when all masks are processed
        self._request_states: Dict[int, Dict] = {}

        # Cache step_map availability check
        self._has_step_map = self.step_map is not None

    def _get_request_slice(
        self,
        forward_batch: ForwardBatch,
        batch_idx: int,
        seq_lens_cpu: List[int],
    ) -> Tuple[int, int, torch.Tensor]:
        """Get request slice info without device sync."""
        offset = sum(seq_lens_cpu[:batch_idx])
        seq_len = seq_lens_cpu[batch_idx]
        req_input_ids = forward_batch.input_ids[offset:offset + seq_len]
        return offset, seq_len, req_input_ids

    def _get_or_init_request_state(
        self,
        req_pool_idx: int,
        forward_batch: ForwardBatch,
        batch_idx: int,
        seq_lens_cpu: List[int],
    ) -> Dict:
        """
        Get or initialize request state.

        Supports:
        - Single request batch
        - Multi-request batch
        - Dynamic batch (requests joining/leaving)
        - Different initial mask counts
        """
        if req_pool_idx not in self._request_states:
            # First time encountering this request, initialize state
            offset, seq_len, req_input_ids = self._get_request_slice(
                forward_batch, batch_idx, seq_lens_cpu
            )

            # Calculate initial mask count (avoid item() if possible)
            mask_eq = req_input_ids == self.mask_id
            initial_mask_count = mask_eq.sum()

            # Get all mask positions (for step_map) - batch CPU transfer
            mask_positions = torch.where(mask_eq)[0].cpu().tolist()

            self._request_states[req_pool_idx] = {
                'step': 0,
                'initial_mask_count': initial_mask_count.item(),
                'initial_mask_positions': mask_positions,
                'step_map': self.step_map,
            }

        return self._request_states[req_pool_idx]

    def _get_target_positions_for_step(
        self,
        state: Dict,
        current_step: int,
        seq_len: int,
    ) -> Optional[List[int]]:
        """
        Get target positions for current step based on step_map.

        Supports:
        - step_map following block_size
        - step_map not following block_size (custom positions)
        - No step_map (returns None for default behavior)
        """
        step_map = state['step_map']

        if step_map is None or current_step not in step_map:
            return None

        target_positions = step_map[current_step]

        # Validate positions are within bounds (vectorized check)
        valid_positions = [pos for pos in target_positions if 0 <= pos < seq_len]

        return valid_positions if valid_positions else None

    def _compute_confidence(
        self,
        logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute predictions and confidence scores (shared logic)."""
        x = torch.argmax(logits, dim=-1)
        probs = F.softmax(logits, dim=-1)
        p = torch.gather(probs, dim=-1, index=x.unsqueeze(-1)).squeeze(-1)
        return x, p

    def _process_with_step_map(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
        req_pool_idx: int,
        batch_idx: int,
        state: Dict,
        seq_lens_cpu: List[int],
    ) -> Tuple[Optional[LogitsProcessorOutput], bool]:
        """
        Process request using step_map.

        Supports:
        - step_map specified positions
        - Position validation and boundary checks
        """
        offset, seq_len, req_input_ids = self._get_request_slice(
            forward_batch, batch_idx, seq_lens_cpu
        )

        current_step = state['step']
        target_positions = self._get_target_positions_for_step(
            state, current_step, seq_len
        )

        if target_positions is None:
            return None, False

        # Vectorized mask creation
        target_tensor = torch.tensor(
            target_positions, device=req_input_ids.device, dtype=torch.long
        )
        valid_mask = (target_tensor < seq_len) & (
            req_input_ids[target_tensor] == self.mask_id
        )

        if not valid_mask.any():
            return None, False

        # Create mask_index using valid positions
        mask_index = torch.zeros(seq_len, dtype=torch.bool, device=req_input_ids.device)
        mask_index[target_tensor[valid_mask]] = True

        # Forward pass
        logits_output, can_run_cuda_graph = model_runner.forward(
            forward_batch, pp_proxy_tensors=None
        )

        # Get this request's logits
        req_logits = logits_output.full_logits[offset:offset + seq_len]

        # Calculate confidence
        x, p = self._compute_confidence(req_logits)

        x = torch.where(mask_index, x, req_input_ids)
        confidence = torch.where(mask_index, p, torch.tensor(-np.inf, device=p.device))

        # Select positions to update
        transfer_index = confidence > self.threshold
        if not transfer_index.any():
            _, select_index = torch.topk(confidence, k=1)
            transfer_index[select_index] = True

        # Vectorized update
        valid_targets = target_tensor[valid_mask]
        update_mask = transfer_index[valid_targets]
        if update_mask.any():
            update_positions = valid_targets[update_mask]
            forward_batch.input_ids[offset + update_positions] = x[update_positions]

        return logits_output, can_run_cuda_graph

    def _process_default(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[
        Union[LogitsProcessorOutput, torch.Tensor], Optional[torch.Tensor], bool
    ]:
        mask_index = forward_batch.input_ids == self.mask_id
        num_masks = mask_index.sum()

        for _ in range(self.block_size):
            mask_index = forward_batch.input_ids == self.mask_id
            num_masks = mask_index.sum()
            if num_masks == 0:
                break

            logits_output, can_run_cuda_graph = model_runner.forward(
                forward_batch, pp_proxy_tensors=None
            )

            x, p = self._compute_confidence(logits_output.full_logits)

            x = torch.where(mask_index, x, forward_batch.input_ids)
            confidence = torch.where(
                mask_index, p, torch.tensor(-np.inf, device=p.device)
            )

            transfer_index = torch.zeros_like(x, dtype=torch.bool, device=x.device)
            _, select_index = torch.topk(confidence, k=1)
            transfer_index[select_index] = True

            forward_batch.input_ids[transfer_index] = x[transfer_index]

    def _check_and_cleanup_finished_requests(
        self,
        forward_batch: ForwardBatch,
        seq_lens_cpu: List[int],
        req_pool_indices_cpu: List[int],
    ):
        """
        Check if requests are finished (all masks processed) and cleanup state.

        This is called automatically after processing to clean up finished requests.
        """
        finished_req_indices = []

        for batch_idx, req_pool_idx in enumerate(req_pool_indices_cpu):
            if req_pool_idx not in self._request_states:
                continue

            offset, seq_len, req_input_ids = self._get_request_slice(
                forward_batch, batch_idx, seq_lens_cpu
            )

            # Check if all masks are processed (avoid item() sync)
            remaining_masks = (req_input_ids == self.mask_id).sum()
            if remaining_masks == 0:
                finished_req_indices.append(req_pool_idx)

        # Cleanup finished requests
        for req_pool_idx in finished_req_indices:
            del self._request_states[req_pool_idx]

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[
        Union[LogitsProcessorOutput, torch.Tensor], Optional[torch.Tensor], bool
    ]:
        """
        Main run logic, supports multi-request batch.

        Supports:
        - Single request batch
        - Multi-request batch (each request processed independently)
        - Dynamic batch (requests joining/leaving)
        - Multi-GPU parallel (TP/PP/DP)
        """
        # Batch CPU transfer once (minimize device sync)
        batch_size = forward_batch.batch_size
        req_pool_indices_cpu = forward_batch.req_pool_indices.cpu().tolist()
        seq_lens_cpu = forward_batch.seq_lens.cpu().tolist()

        # Record start positions (for returning next_token_ids)
        start_positions = []
        for i in range(batch_size):
            start_positions.append(sum(seq_lens_cpu[:i]))

        # Process each request
        all_logits_output = None
        all_can_run_cuda_graph = True

        # Use cached step_map check (avoid repeated runtime checks)
        if self._has_step_map:
            # Process with step_map support
            for batch_idx, req_pool_idx in enumerate(req_pool_indices_cpu):
                # Get or initialize request state
                state = self._get_or_init_request_state(
                    req_pool_idx, forward_batch, batch_idx, seq_lens_cpu
                )

                # Process request (using step_map or default logic)
                logits_output, can_run_cuda_graph = self._process_with_step_map(
                    model_runner, forward_batch, req_pool_idx, batch_idx, state, seq_lens_cpu
                )

                if logits_output is None:
                    # Fall back to default processing for this request
                    # Note: This processes the entire batch, so we only do it once
                    if all_logits_output is None:
                        self._process_default(model_runner, forward_batch)
                        all_logits_output, all_can_run_cuda_graph = model_runner.forward(
                            forward_batch, pp_proxy_tensors=None
                        )

                # Update state
                state['step'] += 1

                # Save result (use the last valid result)
                if logits_output is not None:
                    all_logits_output = logits_output
                    all_can_run_cuda_graph = all_can_run_cuda_graph and can_run_cuda_graph
        else:
            # Process with default logic (no step_map)
            self._process_default(model_runner, forward_batch)
            all_logits_output, all_can_run_cuda_graph = model_runner.forward(
                forward_batch, pp_proxy_tensors=None
            )

        # Check and cleanup finished requests
        self._check_and_cleanup_finished_requests(
            forward_batch, seq_lens_cpu, req_pool_indices_cpu
        )

        # Calculate next_token_ids (merged result for all requests)
        start = min(start_positions) if start_positions else 0
        next_token_ids = forward_batch.input_ids[start:]

        return all_logits_output, next_token_ids, all_can_run_cuda_graph

    def cleanup_request_state(self, req_pool_idx: int):
        """Clean up state for a finished request (called externally if needed)."""
        if req_pool_idx in self._request_states:
            del self._request_states[req_pool_idx]


Algorithm = LowConfidence
