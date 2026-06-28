CUDA_VISIBLE_DEVICES=0 python i2i_tta_acs.py --backbone plip --dataset msmt --use_acs --steps 10 --acs_alpha 0.0 --acs_conf_thresh 0.1 --acs_kmax 5 --acs_groups ubcc,ubcl,lbcc,lbcl,lbct,accessories
CUDA_VISIBLE_DEVICES=0 python i2i_tta_acs.py --backbone plip --dataset market --use_acs --steps 10 --acs_alpha 0.0 --acs_conf_thresh 0.1 --acs_kmax 5 --acs_groups ubcc,ubcl,lbcc,lbcl,lbct,accessories

CUDA_VISIBLE_DEVICES=0 python t2i_tta_acs.py --backbone plip --dataset cuhkpedes --use_acs --steps 10 --acs_alpha 0.0 --acs_conf_thresh 0.1 --acs_kmax 5 --acs_groups age,hair,ubcc,ubcl,lbcc,lbcl,lbct,accessories --hybrid_eval
CUDA_VISIBLE_DEVICES=0 python t2i_tta_acs.py --backbone plip --dataset rstpreid --use_acs --steps 10 --acs_alpha 0.1 --acs_conf_thresh 0.1 --acs_kmax 5 --acs_groups age,hair,ubcc,ubcl,lbcc,lbcl,lbct,accessories --hybrid_eval
