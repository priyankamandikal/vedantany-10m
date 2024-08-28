# Script to run whisper on audio chunks across different gpus
# Update the chunk number and gpu number by adding or removing lines as needed

screen -dmS chunk_0 bash -c "python transcription/run_whisper.py --model-name large-v2 --gpu 0 --chunk 0";
screen -dmS chunk_1 bash -c "python transcription/run_whisper.py --model-name large-v2 --gpu 1 --chunk 1";
screen -dmS chunk_2 bash -c "python transcription/run_whisper.py --model-name large-v2 --gpu 2 --chunk 2";
screen -dmS chunk_3 bash -c "python transcription/run_whisper.py --model-name large-v2 --gpu 3 --chunk 3";
screen -dmS chunk_4 bash -c "python transcription/run_whisper.py --model-name large-v2 --gpu 4 --chunk 4";
screen -dmS chunk_5 bash -c "python transcription/run_whisper.py --model-name large-v2 --gpu 5 --chunk 5";
screen -dmS chunk_6 bash -c "python transcription/run_whisper.py --model-name large-v2 --gpu 6 --chunk 6";
screen -dmS chunk_7 bash -c "python transcription/run_whisper.py --model-name large-v2 --gpu 7 --chunk 7";
