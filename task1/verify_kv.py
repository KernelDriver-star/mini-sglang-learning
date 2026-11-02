import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
from transformers import AutoConfig
from minisgl.core import Req, Batch, Context, SamplingParams
from minisgl.distributed import set_tp_info
from minisgl.models.config import ModelConfig
from minisgl.scheduler.cache import CacheManager
from minisgl.scheduler.table import TableManager
from minisgl.kvcache.mha_pool import MHAKVCache

set_tp_info(rank=0, size=1)
device = torch.device('cuda:0')
page_size, num_pages = 16, 8
table = torch.full((2, 64), -1, dtype=torch.int32, device=device)
manager = CacheManager(num_pages, page_size, table, 'naive')
slots = TableManager(2, table)
# Deliberately permute free pages to verify non-contiguous physical placement.
manager.free_slots = torch.tensor([112, 32, 80, 0, 16, 48, 64, 96], dtype=torch.int32, device=device)
handle = manager.prefix_cache.match_prefix(torch.tensor([], dtype=torch.int32)).cuda_handle
req = Req(torch.arange(35, dtype=torch.int32), slots.allocate(), 0, 4, 1, SamplingParams(max_tokens=4), handle)
batch = Batch([req], phase='prefill')
manager.allocate_paged(batch.reqs)
mapping = table[req.table_idx, :48].cpu().tolist()
expected = list(range(112,128)) + list(range(32,48)) + list(range(80,96))
assert mapping == expected
assert len(manager.free_slots) == 5
assert mapping[34] == 82

pool = MHAKVCache(2, 2, 128, num_pages, page_size, torch.float16, device)
pool._kv_buffer.zero_()
out_loc = table[req.table_idx, :35].clone()
for layer_id in range(2):
    k = torch.full((35, 256), 1.0 + 2*layer_id, dtype=torch.float16, device=device)
    v = torch.full_like(k, 2.0 + 2*layer_id)
    pool.store_kv(k, v, out_loc, layer_id)
torch.cuda.synchronize()
for layer_id in range(2):
    assert torch.all(pool.k_cache(layer_id).view(128,2,128)[out_loc.long()] == 1+2*layer_id)
    assert torch.all(pool.v_cache(layer_id).view(128,2,128)[out_loc.long()] == 2+2*layer_id)
assert torch.all(pool.k_cache(0).view(128,2,128)[83] == 0)

ctx = Context(page_size)
ctx.page_table, ctx.kv_cache = table, pool
with ctx.forward_batch(batch):
    assert ctx.batch is batch
assert ctx._batch is None

req.complete_one()  # prefill has populated 35 KV positions; next step has 1 new token
assert req.extend_len == 1
decode = Batch([req], phase='decode')
manager.allocate_paged(decode.reqs)
assert len(manager.free_slots) == 5  # token 35 still fits in the third page
decode_loc = table[req.table_idx, 35:36]
pool.store_kv(torch.full((1,256),9.,device=device,dtype=torch.float16),
              torch.full((1,256),10.,device=device,dtype=torch.float16),decode_loc,0)
torch.cuda.synchronize()
assert pool.k_cache(0).view(128,2,128)[83,0,0].item() == 9
assert pool.k_cache(0).view(128,2,128)[82,0,0].item() == 1
assert pool.k_cache(1).view(128,2,128)[83,0,0].item() == 0
req.complete_one()
manager.cache_req(req, finished=True)
slots.free(req.table_idx)
assert len(manager.free_slots) == 8
assert len(set(manager.free_slots.cpu().tolist())) == 8
assert slots.available_size == 2
manager.check_integrity()

model_path = next(Path('/home/huang/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots').iterdir())
config = ModelConfig.from_hf(AutoConfig.from_pretrained(model_path, local_files_only=True))
result = {
    'time':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
    'source_commit':subprocess.check_output(['git','-C','/home/huang/mini-sglang-work/mini-sglang','rev-parse','HEAD'],text=True).strip(),
    'gpu':torch.cuda.get_device_name(0), 'torch':torch.__version__,
    'scope':'Actual Mini allocator + CUDA store_cache; synthetic K/V, no model forward or HiCache deployment in this test',
    'config_read_only':{'model':config.model_type,'layers':config.num_layers,'kv_heads':config.num_kv_heads,'head_dim':config.head_dim},
    'page_size':16, 'input_tokens':35, 'allocated_pages':3, 'tail_unused_slots':13,
    'physical_pages':[7,2,5], 'token_34_physical_slot':82,
    'pool_shape':list(pool._kv_buffer.shape),
    'checks':{
        'noncontiguous_page_mapping':'PASS', 'cuda_store_kv_two_layers':'PASS',
        'layer_isolation_and_untouched_slot':'PASS', 'decode_reuses_partial_page':'PASS',
        'decode_write_preserves_history':'PASS', 'context_lifetime':'PASS',
        'free_pages_restored_without_duplicates':'PASS', 'request_slots_restored':'PASS'
    }
}
Path(__file__).with_name('verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(result,ensure_ascii=False,indent=2))
print('PASS: 8 checks; real CUDA KV writes and allocator lifecycle verified')
