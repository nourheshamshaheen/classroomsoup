import argparse, copy, os, re, torch, timm
from torchmetrics import F1Score

from reg_train import get_regular_dataloaders
from curriculum_data import (
    get_data_curriculum_dataloaders,
    get_c2f_dataloaders,
)

def load_state_dict(path, device="cpu"):
    """Load a checkpoint from *path* and strip Lightning prefixes."""
    ckpt = torch.load(path, map_location=device)
    raw  = ckpt.get("state_dict", ckpt)
    return {re.sub(r"^(?:model\.|pl_module\.)+", "", k): v for k, v in raw.items()}



def validate_fn(model, dataloader, device):
    """Compute macro‑F1 on *dataloader* (like Lightning)."""
    model.eval()
    num_classes = getattr(model, "num_classes", 1000)
    f1_metric   = F1Score(task="multiclass", num_classes=num_classes, average="macro")
    with torch.no_grad():
        for imgs, lbls in dataloader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            preds = model(imgs).argmax(dim=1)
            f1_metric.update(preds.cpu(), lbls.cpu())
    return f1_metric.compute().item()



def average_weights(model_dicts):
    """Arithmetic mean of float parameters; ints copied from first dict."""
    num, avg = len(model_dicts), {}
    for k, v in model_dicts[0].items():
        avg[k] = v.clone()
    for d in model_dicts[1:]:
        for k in avg:
            if avg[k].is_floating_point():
                avg[k] += d[k]
    for k in avg:
        if avg[k].is_floating_point():
            avg[k] /= num
    return avg



def greedy_soup(model_paths, model_names, model_scores,
                validate_fn, val_loader, device, model_ctor):
    # 1) sort by solo score
    ranked = sorted(zip(model_names, model_paths),
                    key=lambda x: model_scores[x[0]], reverse=True)
    names, paths = zip(*ranked)

    # 2) seed soup
    first_state      = load_state_dict(paths[0], "cpu")
    chosen_states    = [first_state]
    chosen_names     = [names[0]]
    current_soup     = average_weights(chosen_states)
    current_f1       = model_scores[names[0]]
    print(f"Seed: {names[0]}  F1={current_f1:.4f}\n")

    # 3) iterate
    for name, path in zip(names[1:], paths[1:]):
        candidate = load_state_dict(path, "cpu")
        N         = len(chosen_states)
        temp_soup = {k: (current_soup[k]*N + candidate[k])/(N+1)
                     for k in current_soup}

        tmp_model = model_ctor().to(device)
        tmp_model.load_state_dict(temp_soup, strict=False)
        new_f1    = validate_fn(tmp_model, val_loader, device)
        del tmp_model; torch.cuda.empty_cache()

        print(f"Try {name:25s}  solo={model_scores[name]:.4f}  merged={new_f1:.4f}")
        if new_f1 >= current_f1:
            chosen_states.append(candidate)
            chosen_names.append(name)
            current_soup = average_weights(chosen_states)
            current_f1   = new_f1
            print("  ✔ kept\n")
        else:
            print("  ✘ dropped\n")
        del candidate; torch.cuda.empty_cache()

    print("Greedy soup ingredients:", chosen_names)
    return current_soup



def iterative_greedy_soup(model_paths, model_names, model_scores,
                          validate_fn, val_loader, device,
                          model_ctor, epochs=5):
    # Sort models by descending validation F1 score
    sorted_models = sorted(zip(model_names, model_paths),
                           key=lambda x: model_scores[x[0]], reverse=True)
    names, paths = zip(*sorted_models)

    # Initialize with the best model
    best_model_state = load_state_dict(paths[0], "cpu")
    chosen_states = [best_model_state]
    chosen_names = [names[0]]
    current_soup = average_weights(chosen_states)
    current_f1 = model_scores[names[0]]
    print(f"Seed: {names[0]}  F1={current_f1:.4f}\n")

    for epoch in range(1, epochs + 1):
        print(f"--- Iteration {epoch} ---")
        added = 0
        for name, path in zip(names[1:], paths[1:]):
            candidate_state = load_state_dict(path, "cpu")
            temp_soup = average_weights(chosen_states + [candidate_state])

            temp_model = model_ctor().to(device)
            temp_model.load_state_dict(temp_soup, strict=False)
            new_f1 = validate_fn(temp_model, val_loader, device)
            del temp_model; torch.cuda.empty_cache()

            print(f"Trying {name:25s}  solo={model_scores[name]:.4f}  merged={new_f1:.4f}")
            if new_f1 > current_f1:
                chosen_states.append(candidate_state)
                chosen_names.append(name)
                current_soup = average_weights(chosen_states)
                current_f1 = new_f1
                added += 1
                print("  ✔ added to soup\n")
            else:
                print("  ✘ not added\n")
            del candidate_state; torch.cuda.empty_cache()
        if added == 0:
            print("No improvement in this iteration, stopping.")
            break
    print("Final greedy soup includes:", chosen_names)
    print(f"Final greedy soup F1: {current_f1:.4f}")
    return current_soup




def fisher_weighted_averaging(model_dicts, fisher_mats, lam=None):
    lam = lam or [1.0]*len(model_dicts)
    merged = {}
    for k in model_dicts[0]:
        nume = sum(l*f[k]*m[k] for l,f,m in zip(lam, fisher_mats, model_dicts))
        deno = sum(l*f[k]     for l,f   in zip(lam, fisher_mats))
        stack = torch.stack([m[k] for m in model_dicts])
        merged[k] = torch.where(deno==0, stack.mean(0), nume/deno)
    return merged



def compute_fisher_matrix(model, dataloader, device):
    model.eval()
    fisher, crit, total = {}, torch.nn.CrossEntropyLoss(reduction="sum"), 0
    for n,p in model.named_parameters():
        fisher[n] = torch.zeros_like(p, device=device)
    for imgs, lbls in dataloader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        model.zero_grad(set_to_none=True)
        crit(model(imgs), lbls).backward()
        total += imgs.size(0)
        for n,p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.detach()**2
    for n in fisher:
        fisher[n] /= total
    return fisher


def get_model_ctor(name, num_classes):
    return lambda: timm.create_model(name, num_classes=num_classes)



if __name__ == "__main__":
    P = argparse.ArgumentParser("Incremental model merging")
    # core
    P.add_argument("--technique", choices=["greedy","iterative","fisher"], required=True)
    P.add_argument("--model-name", required=True)
    P.add_argument("--num-classes", type=int, default=1000)
    P.add_argument("--checkpoints", nargs="+", required=True)
    P.add_argument("--device", default="cuda")
    P.add_argument("--output", default="merged_model.pth")
    # scoring
    P.add_argument("--validate-each-checkpoint", action="store_true")
    P.add_argument("--epochs", type=int, default=5)
    # fisher
    P.add_argument("--fisher-paths", nargs="+")
    P.add_argument("--lambdas")
    P.add_argument("--compute-fisher", action="store_true")
    # data / curriculum
    P.add_argument("--curriculum-type", choices=["data","task","None"], required=True)
    P.add_argument("--dataset", default="CIFAR100")
    P.add_argument("--batch-size", type=int, default=64)
    P.add_argument("--task-curr-stage", type=int, choices=[1,2], default=1)
    P.add_argument("--val_indices_path", default="split/cifar100_val_split.npz")
    args = P.parse_args()

    device        = args.device
    model_ctor    = get_model_ctor(args.model_name, args.num_classes)
    model_paths   = args.checkpoints
    model_names   = [os.path.basename(p) for p in model_paths]

    # loaders (reuse saved split)
    if args.curriculum_type == "data":
        class Dummy: current_fraction = 1.0
        _, val_loader, _ = get_data_curriculum_dataloaders(
            args.dataset, args.batch_size, Dummy(),
            model=args.model_name, val_indices_path=args.val_indices_path)
    elif args.curriculum_type == "task":
        _, val_loader = get_c2f_dataloaders(
            args.dataset, args.batch_size, args.task_curr_stage,
            model=args.model_name, val_indices_path=args.val_indices_path)
    else:
        _, val_loader = get_regular_dataloaders(
            args.dataset, args.batch_size,
            model_name=args.model_name, val_indices_path=args.val_indices_path)

    # scores
    model_scores = {}
    if args.technique in ["greedy","iterative"]:
        if args.validate_each_checkpoint:
            for n, p in zip(model_names, model_paths):
                state = load_state_dict(p, "cpu")
                m     = model_ctor().to(device)
                m.load_state_dict(state, strict=False)
                model_scores[n] = validate_fn(m, val_loader, device)
                del m, state; torch.cuda.empty_cache()
        else:
            for n in model_names:
                m = re.search(r"val_f1=([0-9]*\.?[0-9]+)", n)
                if m: model_scores[n] = float(m.group(1))
        print("Scores:", {k: round(v,4) for k,v in model_scores.items()})

    # fisher pre‑load
    if args.technique == "fisher":
        model_dicts = [load_state_dict(p, "cpu") for p in model_paths]
        if args.fisher_paths:
            fisher_mats = [torch.load(fp, map_location="cpu") for fp in args.fisher_paths]
        elif args.compute_fisher:
            class Dummy: current_fraction = 1.0
            tr_loader, _, _ = get_data_curriculum_dataloaders(
                args.dataset, args.batch_size, Dummy(),
                model=args.model_name, val_indices_path=args.val_indices_path)
            fisher_mats = []
            for sd in model_dicts:
                m = model_ctor().to(device)
                m.load_state_dict(sd, strict=False)
                fisher_mats.append(compute_fisher_matrix(m, tr_loader, device))
                del m; torch.cuda.empty_cache()
        else:
            raise ValueError("Need --fisher-paths or --compute-fisher")
        lam = [float(x) for x in args.lambdas.split(",")] if args.lambdas else None
        merged = fisher_weighted_averaging(model_dicts, fisher_mats, lam)

    elif args.technique == "greedy":
        merged = greedy_soup(model_paths, model_names, model_scores,
                             validate_fn, val_loader, device, model_ctor)
    else:
        merged = iterative_greedy_soup(model_paths, model_names, model_scores,
                                               validate_fn, val_loader, device,
                                               model_ctor, epochs=args.epochs)

    torch.save(merged, args.output)
    print("Merged model saved to", args.output)
