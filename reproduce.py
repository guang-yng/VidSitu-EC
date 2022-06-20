import numpy as np
import torch, random
from slowfast.models.video_model_builder import SlowFast
from vidsitu_code.extended_config import CfgProcessor

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def set_seed(s = 0):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)

kwargs = {}
CFP = CfgProcessor("./configs/vsitu_cfg.yml")
cfg = CFP.get_vsitu_default_cfg()
key_maps = CFP.get_key_maps()
cfg = CFP.pre_proc_config(cfg, kwargs)
cfg = CFP.update_from_dict(cfg, kwargs, key_maps)
cfg = CFP.post_proc_config(cfg)
cfg.freeze()

def get_mdl():
    set_seed()
    return SlowFast(cfg.sf_mdl).to('cuda')

def get_inp():
    set_seed(1)
    slow = torch.randn((5, 3, 8, 224, 224,), device="cuda")
    fast = torch.randn((5, 3, 32, 224, 224,), device="cuda")
    return [slow, fast]

loss_fn = torch.nn.CrossEntropyLoss()
target = torch.randint(0, 400, (5, ), device="cuda")

mdl1 = get_mdl()
inp1 = get_inp()
out1 = mdl1.s1(inp1)
out1 = mdl1.s1_fuse(out1)
out1 = mdl1.s2(out1)
out1 = mdl1.s2_fuse(out1)
for pathway in range(mdl1.num_pathways):
    pool = getattr(mdl1, "pathway{}_pool".format(pathway))
    out1[pathway] = pool(out1[pathway])
out1 = mdl1.s3(out1)
out1 = mdl1.s3_fuse(out1)
out1 = mdl1.s4(out1)
out1 = mdl1.s4_fuse(out1)
out1 = mdl1.s5(out1)
out1 = mdl1.head(out1)

l1 = loss_fn(out1, target)
l1.backward()
grad1 = {}
for k, v in mdl1.named_parameters():
    grad1[k] = v.grad

mdl2 = get_mdl()
inp2 = get_inp()
out2 = mdl2.s1(inp2)
out2 = mdl2.s1_fuse(out2)
out2 = mdl2.s2(out2)
out2 = mdl2.s2_fuse(out2)
for pathway in range(mdl2.num_pathways):
    pool = getattr(mdl2, "pathway{}_pool".format(pathway))
    out2[pathway] = pool(out2[pathway])
out2 = mdl2.s3(out2)
out2 = mdl2.s3_fuse(out2)
out2 = mdl2.s4(out2)
out2 = mdl2.s4_fuse(out2)
out2 = mdl2.s5(out2)
out2 = mdl2.head(out2)
l2 = loss_fn(out2, target)
l2.backward()
grad2 = {}
for k, v in mdl2.named_parameters():
    grad2[k] = v.grad


print((inp1[0] == inp2[0]).all(), (inp1[1] == inp2[1]).all())
print((out1 == out2).all())
print(l1 == l2)
for k in grad1.keys():
    b = grad1[k] == grad2[k]
    if not isinstance(b, bool):
        b = b.all()
    if not b:
        g1 = grad1[k].flatten()
        g2 = grad2[k].flatten()
        for i in range(len(g1)):
            if g1[i] != g2[i]:
                print(k, i, g1[i].item(), g2[i].item())
                break