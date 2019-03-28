mkdir Criteo
python ./Dataprocess/Criteo/preprocess.py
python ./Dataprocess/Kfold_split/stratifiedKfold.py
python ./Dataprocess/Criteo/scale.py
