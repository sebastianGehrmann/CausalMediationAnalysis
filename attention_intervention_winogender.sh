# Note: these scripts are somewhat computationally expensive, so you may want to perform a subset only
#
# Perform attention intervention using bergsma statistics
#
python winogender_attn_intervention.py --gpt2-version distilgpt2 --do-filter True --stat bergsma &&
python winogender_attn_intervention.py --gpt2-version distilgpt2 --do-filter False --stat bergsma &&
python winogender_attn_intervention.py --gpt2-version gpt2 --do-filter True --stat bergsma &&
python winogender_attn_intervention.py --gpt2-version gpt2 --do-filter False --stat bergsma &&
python winogender_attn_intervention.py --gpt2-version gpt2-medium --do-filter True --stat bergsma &&
python winogender_attn_intervention.py --gpt2-version gpt2-medium --do-filter False --stat bergsma &&
python winogender_attn_intervention.py --gpt2-version gpt2-large --do-filter True --stat bergsma &&
python winogender_attn_intervention.py --gpt2-version gpt2-large --do-filter False --stat bergsma &&
python winogender_attn_intervention.py --gpt2-version gpt2-xl --do-filter True --stat bergsma &&
python winogender_attn_intervention.py --gpt2-version gpt2-xl --do-filter False --stat bergsma &&
#
# Perform attention intervention using bls statistics
#
python winogender_attn_intervention.py --gpt2-version distilgpt2 --do-filter True --stat bls &&
python winogender_attn_intervention.py --gpt2-version distilgpt2 --do-filter False --stat bls &&
python winogender_attn_intervention.py --gpt2-version gpt2 --do-filter True --stat bls &&
python winogender_attn_intervention.py --gpt2-version gpt2 --do-filter False --stat bls &&
python winogender_attn_intervention.py --gpt2-version gpt2-medium --do-filter True --stat bls &&
python winogender_attn_intervention.py --gpt2-version gpt2-medium --do-filter False --stat bls &&
python winogender_attn_intervention.py --gpt2-version gpt2-large --do-filter True --stat bls &&
python winogender_attn_intervention.py --gpt2-version gpt2-large --do-filter False --stat bls &&
python winogender_attn_intervention.py --gpt2-version gpt2-xl --do-filter True --stat bls &&
python winogender_attn_intervention.py --gpt2-version gpt2-xl --do-filter False --stat bls &&
#
# Perform attention intervention using random model
#
python winogender_attn_intervention.py --gpt2-version gpt2 --do-filter True --stat bls --random-weights True
