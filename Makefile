install:
	echo "python3 -m venv rc2 && source rc2/bin/activate && pip3 install -r requirements.txt"

data:
	kaggle competitions download -c recsys-20212-rc2
	unzip recsys-20212-rc2.zip -d data

run:
	python3 main.py data/ratings.jsonl data/content.jsonl data/targets.csv > predictions.csv

submit:
	kaggle competitions submit -c recsys-20212-rc2 -f predictions.csv -m "$(date)"