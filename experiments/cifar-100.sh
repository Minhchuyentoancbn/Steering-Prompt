# bash experiments/cifar-100.sh
# experiment settings
DATASET=cifar-100
N_CLASS=100

# save directory
OUTDIR=outputs/${DATASET}/10-task

# hard coded inputs
GPUID='0 1'
CONFIG=configs/cifar-100_prompt.yaml
REPEAT=1
OVERWRITE=0

###############################################################

# process inputs
mkdir -p $OUTDIR

# # CODA-P
# #
# # prompt parameter args:
# #    arg 1 = prompt component pool size
# #    arg 2 = prompt length
# #    arg 3 = ortho penalty loss weight - with updated code, now can be 0!
# python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#     --learner_type prompt --learner_name CODAPrompt \
#     --prompt_param 100 8 0.0 \
#     --log_dir ${OUTDIR}/coda-p

# # DualPrompt
# #
# # prompt parameter args:
# #    arg 1 = e-prompt pool size (# tasks)
# #    arg 2 = e-prompt pool length
# #    arg 3 = g-prompt pool length
# python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#     --learner_type prompt --learner_name DualPrompt \
#     --prompt_param 10 20 6 \
#     --log_dir ${OUTDIR}/dual-prompt

# # L2P++
# #
# # prompt parameter args:
# #    arg 1 = e-prompt pool size (# tasks)
# #    arg 2 = e-prompt pool length
# #    arg 3 = -1 -> shallow, 1 -> deep
# python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#     --learner_type prompt --learner_name L2P \
#     --prompt_param 30 20 -1 \
#     --log_dir ${OUTDIR}/l2p++


# CPP
#
# prompt parameter args:
#    arg 1 = e-prompt pool size (# tasks)
#    arg 2 = number of centroids
#    arg 3 = number of nearest neighbors
#    arg 4 = number of classes
python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type prompt --learner_name CPP \
    --prompt_param 10 5 3 100 \
    --log_dir ${OUTDIR}/cpp


# python -u run.py --config configs/cifar-100_prompt.yaml --gpuid 0 --repeat 1 --overwrite 0 --learner_type prompt --learner_name CPP --prompt_param 10 5 3 100 --log_dir outputs/cifar-100/10-task/cpp