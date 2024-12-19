import math


# learning rate decay scheduler (cosine with warmup)
def get_lr(it, cfg):
    # 1) linear warmup for warmup_iters steps
    if it < cfg.warmup_iters:
        return cfg.init_lr + (cfg.base_lr - cfg.init_lr) * it / cfg.warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > cfg.lr_decay_iters:
        return cfg.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - cfg.warmup_iters) / (cfg.lr_decay_iters - cfg.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return cfg.min_lr + coeff * (cfg.base_lr - cfg.min_lr)


if __name__=='__main__':

    from easydict import EasyDict
    cfg = EasyDict(dict(
        init_lr=1e-4,
        base_lr=1e-2,
        min_lr=1e-4,
        warmup_iters=10,
        lr_decay_iters=50,
    ))

    for it in range(60):
        _lr = get_lr(it, cfg)
        print(f'epoch: {it}, lr: {_lr}')

