import torch
import os

'''
This file uses a trained model to sample/generate robots using final_dl method of GFN.
The generated robot xml files are saved into the modified log_dir folder which is available from the hps of saved_state of the trained model.
The folder path of these generated robots can be used in evaluate_robots.py to evaluate sampled robots performance. 
'''

def cycle(it):
        while True:
            for i in it:
                yield i


def main():

    exp = 'CA'

    if exp == 'CA':
        from gflownet_costaware.models.graph_transformer import GraphTransformerGFN
        from gflownet_costaware.envs.frag_mol_env import FragMolBuildingEnvContext
        from gflownet_costaware.tasks.seh_frag import SEHFragTrainer
    else:
        from gflownet_old.models.graph_transformer import GraphTransformerGFN
        from gflownet_old.envs.frag_mol_env import FragMolBuildingEnvContext
        from gflownet_old.tasks.seh_frag import SEHFragTrainer


    # Required Inputs based on the characteristics of the run
    lower_bound = 400
    upper_bound = 400
    step_size = 100
    path = "/home/knagiredla/gfn_archive/gfn_current/gfn_fixed_comp/logs/exp_orig1_75_300k_gsca_flat_3_1725246076/policies"

    for idx in range(lower_bound, upper_bound+1, step_size):
        
        # print("idx", idx)
        
        model_path = path + f"/model_state_{idx}.pt"

        env_ctx = FragMolBuildingEnvContext(max_frags=9, num_cond_dim=32)

        # Define an instance of YourModelClass (assuming it's the class of self.model)
        model = GraphTransformerGFN(env_ctx, num_emb=128, num_layers=3)

        # Load the model state from the saved file
        saved_state = torch.load(model_path)  # Replace with the actual file path

        print("saved_state", saved_state)

        # Update the model's state with the loaded state
        model.load_state_dict(saved_state["models_state_dict"][0])  # Assuming the state was saved within a list

        # Access other saved information if needed
        # Eg. model_state_dict = saved_state["models_state_dict"]
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        hps = saved_state["hps"]
        it = saved_state["step"]
        new_hps = hps

        new_hps['log_dir'] = new_hps['xml_path'] + f'/gen_{it}_steps'

        if not os.path.isdir(new_hps['log_dir']):
            os.mkdir(new_hps['log_dir'])

        trainer2 = SEHFragTrainer(hps=new_hps, device=device)
        # print("THIS IS TRAINER", trainer2)
        # print(type(trainer2))
        final_dl = trainer2.build_final_data_loader()

        for it, batch in zip(
                            range(new_hps["num_training_steps"], new_hps["num_training_steps"] + 1), 
                            cycle(final_dl),
                            ):
                                pass
        
        

if __name__ == "__main__":
    main()


