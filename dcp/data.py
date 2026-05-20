import torch

from dcp.runtime import get_dist_info, print0


class DataLoader:
    """Pre-tokenized chunk dataloader. Yields (inputs, targets, epoch) forever."""

    def __init__(self, filepath, B, T, device="cuda"):
        data = torch.load(filepath, weights_only=True)
        chunks = data["chunks"]
        valid_counts = data["valid_counts"]
        file_B = data["batch_size"]
        sequence_size = data["sequence_size"]
        assert sequence_size == T + 1, f"Data sequence_size {sequence_size} != T+1={T+1}"

        # Gather all valid sequences into one tensor
        all_seqs = []
        for chunk, vc in zip(chunks, valid_counts):
            rows = chunk.view(file_B, sequence_size)[:vc]
            all_seqs.append(rows)
        all_seqs = torch.cat(all_seqs, dim=0).long()  # (N, T+1)

        # DDP sharding: each rank gets every world_size-th batch
        _, rank, _, world_size = get_dist_info()
        seqs_per_step = B * world_size
        num_steps = len(all_seqs) // seqs_per_step
        usable = num_steps * seqs_per_step
        all_seqs = all_seqs[:usable].view(num_steps, world_size, B, sequence_size)

        self.rank_data = all_seqs[:, rank].contiguous()  # (num_steps, B, T+1)
        self.num_steps = num_steps
        self.total_tokens = usable * T  # trainable tokens across all ranks
        self.device = device
        self.pos = 0
        self.epoch = 1

    def __iter__(self):
        return self

    def _shuffle(self):
        """Shuffle batch order for the new epoch, consistent across ranks."""
        g = torch.Generator()
        g.manual_seed(self.epoch)
        perm = torch.randperm(self.num_steps, generator=g)
        self.rank_data = self.rank_data[perm]

    def __next__(self):
        if self.pos >= self.num_steps:
            self.pos = 0
            self.epoch += 1
            print0(f"Starting epoch {self.epoch}")
            self._shuffle()
        batch = self.rank_data[self.pos].to(self.device, non_blocking=True)
        self.pos += 1
        return batch[:, :-1].contiguous(), batch[:, 1:].contiguous(), self.epoch


def _assert_loader_vocab_compatible(loader, max_token_id, split_name):
    if loader.rank_data.numel() == 0:
        return
    observed_max = int(loader.rank_data.max().item())
    if observed_max > max_token_id:
        raise ValueError(
            f"{split_name} data contains token id {observed_max}, but this model only supports token ids <= {max_token_id}. "
            "Regenerate the packed dataset with the same tokenizer used by this trainer."
        )
