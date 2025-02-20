@echo off

REM Run Python script relative to the batch file location
python train.py config/train_testgpt.py --init_from=resume --device=cuda --out_dir=out-gpt-test-fixed_latest 
