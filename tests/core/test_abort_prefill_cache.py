"""Exercise in-flight prefill aborts with real radix-cache and table accounting."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import minisgl.core as core
import minisgl.distributed.info as dist_info
import pytest
import torch
from minisgl.core import Batch, Req, SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.message import AbortBackendMsg
from minisgl.scheduler.cache import CacheManager
from minisgl.scheduler.decode import DecodeManager
from minisgl.scheduler.prefill import PrefillManager
from minisgl.scheduler.scheduler import Scheduler
from minisgl.scheduler.table import TableManager
from minisgl.scheduler.utils import PendingReq


def make_scheduler(monkeypatch, page_size, device):
    monkeypatch.setattr(core, "_GLOBAL_CTX", core.Context(page_size=page_size))
    monkeypatch.setattr(dist_info, "_TP_INFO", DistributedInfo(0, 1))
    scheduler = Scheduler.__new__(Scheduler)
    page_table = torch.zeros((4, 32), dtype=torch.int32, device=device)
    scheduler.table_manager = TableManager(4, page_table)
    scheduler.cache_manager = CacheManager(32, page_size, page_table, "radix")
    scheduler.decode_manager = DecodeManager(page_size)
    scheduler.prefill_manager = PrefillManager(
        scheduler.cache_manager, scheduler.table_manager, scheduler.decode_manager
    )
    scheduler.finished_reqs = set()
    scheduler.pending_aborts = set()
    scheduler.eos_token_id = 99
    scheduler.engine = SimpleNamespace(stream=Mock())
    scheduler.stream = Mock()
    scheduler.engine_stream_ctx = nullcontext()
    scheduler.receive_msg = Mock(return_value=[])
    scheduler.send_result = Mock()
    scheduler._schedule_next_batch = Mock(return_value=None)
    return scheduler


def make_prefilled_req(scheduler, uid):
    # Set up admission and the completed device step without loading a model or
    # requiring CUDA pinned memory. Cache matching, locking and allocation are real.
    cm = scheduler.cache_manager
    tokens = torch.arange(10, 19, dtype=torch.int32)
    sampling = SamplingParams(max_tokens=8)
    pending = PendingReq(uid, tokens, sampling)
    handle = cm.match_req(pending).cuda_handle
    cm.lock(handle)
    row = scheduler.table_manager.allocate()
    if handle.cached_len:
        cm.page_table[row, : handle.cached_len] = handle.get_matched_indices()
    pages_needed = (len(tokens) + cm.page_size - 1) // cm.page_size
    pages_needed -= handle.cached_len // cm.page_size
    allocated = cm._page_to_token(cm._allocate(pages_needed))
    cm.page_table[row, handle.cached_len : handle.cached_len + len(allocated)] = allocated
    req = Req(tokens, row, handle.cached_len, 8, uid, sampling, handle)
    req.complete_one()
    scheduler.decode_manager.filter_reqs([req])
    return req


@pytest.mark.parametrize("page_size", [1, 4])
@pytest.mark.parametrize("cache_state", ["miss", "hit", "shared_hit"])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
        ),
    ],
)
def test_abort_inflight_prefill_preserves_cache_pages(monkeypatch, page_size, cache_state, device):
    scheduler = make_scheduler(monkeypatch, page_size, device)
    cm = scheduler.cache_manager
    peer_handle = None
    if cache_state != "miss":
        warmup = make_prefilled_req(scheduler, uid=0)
        scheduler.decode_manager.remove_req(warmup)
        scheduler._free_req_resources(warmup)
        cm.check_integrity()
        if cache_state == "shared_hit":
            peer_handle = cm.prefix_cache.match_prefix(warmup.input_ids[:-1]).cuda_handle
            cm.lock(peer_handle)

    req = make_prefilled_req(scheduler, uid=1)
    assert bool(req.cache_handle.cached_len) == (cache_state != "miss")
    rows_during_schedule = []

    def schedule_next():
        rows_during_schedule.append(scheduler.table_manager.available_size)
        assert not scheduler.decode_manager.runnable
        return None

    scheduler._schedule_next_batch.side_effect = schedule_next
    scheduler.receive_msg.return_value = [AbortBackendMsg(uid=req.uid)]
    completed = Mock()
    last_data = (
        SimpleNamespace(batch=Batch([req], "prefill")),
        (None, torch.tensor([20], dtype=torch.int32), completed),
    )

    scheduler.overlap_loop(last_data)

    cm.check_integrity()
    # An abort must not make the old row available while its prefill is in flight.
    assert rows_during_schedule == [3]
    completed.synchronize.assert_called_once()
    assert scheduler.table_manager.available_size == 4
    assert not scheduler.pending_aborts
    if peer_handle is not None:
        assert peer_handle.node.ref_count == 1
        assert cm.prefix_cache.size_info.protected_size == peer_handle.cached_len
        cm.unlock(peer_handle)
    assert cm.prefix_cache.size_info.protected_size == 0
    cm.check_integrity()

    # An already-aborted request must not free anything a second time.
    scheduler.overlap_loop(None)
    assert scheduler.table_manager.available_size == 4
    cm.check_integrity()

    # Reclaim the entire pool, including cached pages. Counts alone would miss a
    # cached page also being on the free list if another page were leaked.
    all_pages = cm._allocate(cm.num_pages)
    assert sorted(all_pages.tolist()) == list(range(0, cm.num_pages * page_size, page_size))
    assert cm.prefix_cache.size_info.total_size == 0
    cm._free(cm._page_to_token(all_pages))
    cm.check_integrity()
