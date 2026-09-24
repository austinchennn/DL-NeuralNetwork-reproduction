import math

from config import Config
from dataset import get_corpus
from engine import configure_optimizer, cosine_lr, estimate_loss, train_step
from model import GPT
from utils import get_device, parse_config, save_checkpoint, set_seed


def main():
    cfg = parse_config(Config)
    set_seed(cfg.seed)
    device = get_device()

    corpus = get_corpus(cfg)
    vocab = len(corpus.tokenizer)
    model = GPT(vocab, cfg.block_size, cfg.n_layer, cfg.n_head, cfg.n_embd, cfg.dropout).to(device)
    print(f"vocab {vocab} | GPT params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M "
          f"| 随机初始化的期望损失 ln({vocab}) = {math.log(vocab):.3f}")
    optimizer = configure_optimizer(model, cfg.lr, cfg.weight_decay)

    best_val = float("inf")
    for it in range(cfg.max_iters + 1):
        if it % cfg.eval_interval == 0:
            losses = estimate_loss(model, corpus, cfg.batch_size, cfg.eval_iters, device)
            print(f"iter {it:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")
            if losses["val"] < best_val:
                best_val = losses["val"]
                save_checkpoint(cfg.ckpt_path, model, cfg, itos=corpus.tokenizer.itos)
        if it == cfg.max_iters:
            break
        for group in optimizer.param_groups:
            group["lr"] = cosine_lr(it, cfg.lr, cfg.min_lr, cfg.warmup_iters, cfg.max_iters)
        train_step(model, corpus, optimizer, cfg.batch_size, device, cfg.grad_clip)
    print(f"best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
