@echo off
set CURRENT_DIR=%cd%
set BERT_BASE_DIR=%CURRENT_DIR%\prev_trained_model\bert-base-chinese
set DATA_DIR=%CURRENT_DIR%\datasets
set OUTPUT_DIR=%CURRENT_DIR%\outputs
set TASK_NAME=cner

python.exe run_ner_span.py ^
  --model_type=bert ^
  --model_name_or_path=%BERT_BASE_DIR% ^
  --task_name=%TASK_NAME% ^
  --do_train ^
  --do_eval ^
  --do_adv ^
  --do_lower_case ^
  --loss_type=ce ^
  --data_dir=%DATA_DIR%\%TASK_NAME%\ ^
  --train_max_seq_length=128 ^
  --eval_max_seq_length=512 ^
  --per_gpu_train_batch_size=8 ^
  --per_gpu_eval_batch_size=16 ^
  --learning_rate=2e-5 ^
  --num_train_epochs=4.0 ^
  --logging_steps=-1 ^
  --save_steps=-1 ^
  --output_dir=%OUTPUT_DIR%\%TASK_NAME%_output\ ^
  --overwrite_output_dir ^
  --seed=42

pause 
::
::python run_ner_span.py --model_type=bert ^
::--model_name_or_path=D:\25new\yiyong-group\BERT-NER-Pytorch\prev_trained_model\bert-base-chinese ^
::--task_name=cner --do_train --do_eval --data_dir=D:\25new\yiyong-group\BERT-NER-Pytorch\datasets\cner\ ^
::--output_dir=D:\25new\yiyong-group\BERT-NER-Pytorch\outputs\cner_output\ --max_seq_length=128 --num_train_epochs=4 ^
::--per_gpu_train_batch_size=8 --per_gpu_eval_batch_size=16 --gradient_accumulation_steps=3 --learning_rate=2e-5 ^
::--crf_learning_rate=5e-5 --weight_decay=0.01 --warmup_proportion=0.1 --do_lower_case --seed=42 ^
::--markup=bios --loss_type=ce
