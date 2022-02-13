import torch

from .misc import config_to_string


def save_snapshot(file, config, epoch, last_score, best_score, global_step, **kwargs):
    data = {
        "config": config_to_string(config),
        "state_dict": dict(kwargs),
        "training_meta": {
            "epoch": epoch,
            "last_score": last_score,
            "best_score": best_score,
            "global_step": global_step
        }
    }
    torch.save(data, file)


def pre_train_from_snapshots(model, snapshots, multiple_models, first_instance_model, only_mfns, modules, rank):
    for snapshot in snapshots:
        if ":" in snapshot:
            module_name, snapshot = snapshot.split(":")
        else:
            module_name = None

        snapshot = torch.load(snapshot, map_location="cpu")
        
        if module_name is None:
            _load_pretraining_dict(getattr(model, 'body'), snapshot)
        else:
            if rank == 0:
                print(module_name, modules)
            if module_name in modules:
                print("HELLLLOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO", multiple_models, first_instance_model, only_mfns)
                if multiple_models and module_name == "flow_head":
                    if not only_mfns:
                        # Load only the decoder part of the model
                        filtered_dict = {k: v for k, v in snapshot['state_dict'].items() if (k.split(".")[2].startswith("cost_bn") or k.split(".")[2].startswith("Decoder"))}
                        renamed_dict = {}
                        prefix = "flow_head"
                        for k, v in filtered_dict.items():
                            tokens = k.split(".")[2:]
                            renamed_key = "{}.{}".format(prefix, ".".join(tokens))
                            renamed_dict[renamed_key] = v
                        snapshot['state_dict'] = renamed_dict
                        _load_pretraining_dict(getattr(model, module_name), snapshot['state_dict'])
                    else:
                        print("Loading only MFNS from the flow_head!")
                        # Load only the MFNS part of the model
                        filtered_dict = {k: v for k, v in snapshot['state_dict']['flow_head'].items()}
                        renamed_dict = {}
                        prefix = "mfn_s"
                        for k, v in filtered_dict.items():
                            renamed_key = "{}.{}".format(prefix, k)
                            renamed_dict[renamed_key] = v
                        snapshot['state_dict']['flow_head'] = renamed_dict
                        print("*********************************** Renaming and Loading {} layers: ***********************************".format(module_name))
                        for key in snapshot['state_dict']['flow_head'].keys():
                            print(key)
                        _load_pretraining_dict(getattr(model, module_name), snapshot['state_dict']['flow_head'])

                elif first_instance_model and module_name == "flow_head":
                    # for k, v in snapshot['state_dict']['flow_head'].items():
                    #     print("Hello111111", k)
                    filtered_dict = {k: v for k, v in snapshot['state_dict']['flow_head'].items()}
                    renamed_dict = {}
                    prefix = "mfn"
                    for k, v in filtered_dict.items():
                        renamed_key = "{}.{}".format(prefix, k)
                        renamed_dict[renamed_key] = v
                    snapshot['state_dict']['flow_head'] = renamed_dict
                    print("*********************************** Renaming and Loading {} layers: ***********************************".format(module_name))
                    for key in snapshot['state_dict']['flow_head'].keys():
                        print(key)
                    _load_pretraining_dict(getattr(model, module_name), snapshot['state_dict']['flow_head'])
                else:
                    state_dict = snapshot['state_dict']
                    if rank == 0:
                        print("*********************************** Loading {} layers: ***********************************".format(module_name))
                        for key in state_dict[module_name].keys():
                            print(key)
                    _load_pretraining_dict(getattr(model, module_name), state_dict[module_name])


            else:
                raise ValueError("Unrecognized network module {}".format(module_name))


def resume_from_snapshot(model, snapshot, modules):
    snapshot = torch.load(snapshot, map_location="cpu")
    state_dict = snapshot["state_dict"]

    for module in modules:
        if module in state_dict:
            _load_pretraining_dict(getattr(model, module), state_dict[module])
        else:
            raise KeyError("The given snapshot does not contain a state_dict for module '{}'".format(module))

    return snapshot


def _load_pretraining_dict(model, state_dict):
    """Load state dictionary from a pre-training snapshot

    This is an even less strict version of `model.load_state_dict(..., False)`, which also ignores parameters from
    `state_dict` that don't have the same shapes as the corresponding ones in `model`. This is useful when loading
    from pre-trained models that are trained on different datasets.

    Parameters
    ----------
    model : torch.nn.Model
        Target model
    state_dict : dict
        Dictionary of model parameters
    """
    model_sd = model.state_dict()

    for k, v in model_sd.items():
        if k in state_dict:
            if v.shape != state_dict[k].shape:
                del state_dict[k]
 
    model.load_state_dict(state_dict, False)
