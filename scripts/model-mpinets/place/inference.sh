CKPT_PATH=$1

python inference_mpinets.py hydra/job_logging=none hydra/hydra_logging=none \
                exp_dir=${CKPT_PATH} \
                model=mpinets_mk \
                task=mk_mpinets_place \
                task.environment.sim_gui=false \
                task.environment.viz=false \